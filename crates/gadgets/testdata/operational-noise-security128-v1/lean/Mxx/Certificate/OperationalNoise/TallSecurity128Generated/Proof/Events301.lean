import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events301

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact77056RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43825⟩⟩]⟩, (1)⟩]

theorem exact77056RawTermsValid :
    exact77056RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77056 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43825⟩⟩) exact77056RawTerms .large 77055 .exactZero (none)

def event77057 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44365⟩⟩) 0 ⟨43825⟩ 77056

def event77058 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44365⟩⟩) (.authority (.operator))

def exact77059RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44365⟩⟩]⟩, (1)⟩]

theorem exact77059RawTermsValid :
    exact77059RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77059 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44365⟩⟩) exact77059RawTerms (.finite 8192) 77058 .exactZero (none)

def event77060 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event77061 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event77062 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44090⟩⟩) 0 ⟨42620⟩ 77048

def event77063 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44090⟩⟩) 1 ⟨136⟩ 77061

def event77064 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44090⟩⟩) (.sum [.predecessor 0 77062 .coefficient, .predecessor 1 77063 .coefficient])

def event77065 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44090⟩⟩) (.finite 2704)

def event77066 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44091⟩⟩) 0 ⟨44090⟩ 77065

def event77067 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44091⟩⟩) (.identity (.predecessor 0 77066 .coefficient))

def exact77068RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], []⟩, (1)⟩]

theorem exact77068RawTermsValid :
    exact77068RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77068 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44091⟩⟩) exact77068RawTerms (.finite 2704) 77067 .exactZero (none)

def event77069 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact77070RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact77070RawTermsValid :
    exact77070RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77070 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact77070RawTerms .large 77069 .exactZero (none)

def event77071 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44092⟩⟩) 0 ⟨6908⟩ 77070

def event77072 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44092⟩⟩) 1 ⟨44091⟩ 77068

def event77073 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44092⟩⟩) (.product (.predecessor 0 77071 .coefficient) (.predecessor 1 77072 .coefficient) (⟨false, false, none, none, none⟩))

def event77074 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44092⟩⟩, .operator (⟨77070, 0⟩, ⟨77068, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact77075RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact77075RawTermsValid :
    exact77075RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77075 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44092⟩⟩) exact77075RawTerms .large 77073 .exactZero (none)

def event77076 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event77077 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event77078 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 77052

def event77079 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact77080RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact77080RawTermsValid :
    exact77080RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77080 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact77080RawTerms .large 77079 .exactZero (none)

def event77081 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7283⟩⟩) 0 ⟨7178⟩ 77080

def event77082 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7283⟩⟩) (.identity (.predecessor 0 77081 .coefficient))

def exact77083RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩, (1)⟩]

theorem exact77083RawTermsValid :
    exact77083RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77083 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7283⟩⟩) exact77083RawTerms .large 77082 .exactZero (none)

def event77084 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9559⟩⟩) 0 ⟨7283⟩ 77083

def event77085 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9559⟩⟩) (.authority (.operator))

def exact77086RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (1)⟩]

theorem exact77086RawTermsValid :
    exact77086RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77086 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9559⟩⟩) exact77086RawTerms (.finite 8192) 77085 .exactZero (none)

def event77087 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9560⟩⟩) 0 ⟨9559⟩ 77086

def event77088 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9560⟩⟩) 1 ⟨2370⟩ 77077

def event77089 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9560⟩⟩) (.scale (.predecessor 0 77087 .coefficient) (.value (.predecessor 1 77088 .coefficient)))

def exact77090RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (1)⟩]

theorem exact77090RawTermsValid :
    exact77090RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77090 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9560⟩⟩) exact77090RawTerms (.finite 8192) 77089 .exactZero (none)

def event77091 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7300⟩⟩) 0 ⟨7178⟩ 77080

def event77092 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7300⟩⟩) (.identity (.predecessor 0 77091 .coefficient))

def exact77093RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩]⟩, (1)⟩]

theorem exact77093RawTermsValid :
    exact77093RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77093 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7300⟩⟩) exact77093RawTerms .large 77092 .exactZero (none)

def event77094 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9561⟩⟩) 0 ⟨7300⟩ 77093

def event77095 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9561⟩⟩) 1 ⟨9560⟩ 77090

def event77096 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9561⟩⟩) (.product (.predecessor 0 77094 .coefficient) (.predecessor 1 77095 .coefficient) (⟨false, false, none, none, none⟩))

def event77097 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9561⟩⟩, .operator (⟨77093, 0⟩, ⟨77090, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (1)⟩)

def exact77098RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (1)⟩]

theorem exact77098RawTermsValid :
    exact77098RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77098 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9561⟩⟩) exact77098RawTerms .large 77096 .exactZero (none)

def event77099 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44093⟩⟩) 0 ⟨9561⟩ 77098

def event77100 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44093⟩⟩) 1 ⟨44092⟩ 77075

def event77101 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44093⟩⟩) (.sum [.predecessor 0 77099 .coefficient, .predecessor 1 77100 .coefficient])

def exact77102RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77102RawTermsValid :
    exact77102RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77102 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44093⟩⟩) exact77102RawTerms .large 77101 .exactZero (none)

def event77103 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44368⟩⟩) 0 ⟨44093⟩ 77102

def event77104 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44368⟩⟩) 1 ⟨44365⟩ 77059

def event77105 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44368⟩⟩) (.product (.predecessor 0 77103 .coefficient) (.predecessor 1 77104 .coefficient) (⟨false, false, none, none, none⟩))

def event77106 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44368⟩⟩, .operator (⟨77102, 0⟩, ⟨77059, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩]⟩, (1)⟩)

def event77107 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44368⟩⟩, .operator (⟨77102, 1⟩, ⟨77059, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩]⟩, (-1)⟩)

def event77108 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44368⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44365⟩⟩) ⟨43825⟩ 77056)

def event77109 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44368⟩⟩, .relation 77108 0, ⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], [⟨.program ⟨257⟩, ⟨43825⟩⟩]⟩, (-1)⟩)

def exact77110RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], [⟨.program ⟨257⟩, ⟨43825⟩⟩]⟩, (-1)⟩]

theorem exact77110RawTermsValid :
    exact77110RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77110 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44368⟩⟩) exact77110RawTerms .large 77105 .exactZero (none)

def event77111 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42836⟩⟩) 0 ⟨42620⟩ 77048

def event77112 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42836⟩⟩) (.authority (.programFamilyFact))

def exact77113RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], []⟩, (1)⟩]

theorem exact77113RawTermsValid :
    exact77113RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77113 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42836⟩⟩) exact77113RawTerms (.finite 52) 77112 .exactZero (none)

def event77114 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42838⟩⟩) 0 ⟨6908⟩ 77070

def event77115 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42838⟩⟩) 1 ⟨42836⟩ 77113

def event77116 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42838⟩⟩) (.product (.predecessor 0 77114 .coefficient) (.predecessor 1 77115 .coefficient) (⟨false, true, none, none, some 1⟩))

def event77117 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42838⟩⟩, .operator (⟨77070, 0⟩, ⟨77113, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact77118RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact77118RawTermsValid :
    exact77118RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77118 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42838⟩⟩) exact77118RawTerms .large 77116 .exactZero (none)

def event77119 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7194⟩⟩) 0 ⟨7177⟩ 77052

def event77120 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7194⟩⟩) (.authority (.operator))

def exact77121RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩]

theorem exact77121RawTermsValid :
    exact77121RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77121 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7194⟩⟩) exact77121RawTerms .large 77120 .exactZero (none)

def event77122 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42839⟩⟩) 0 ⟨7194⟩ 77121

def event77123 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42839⟩⟩) 1 ⟨42838⟩ 77118

def event77124 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42839⟩⟩) (.sum [.predecessor 0 77122 .coefficient, .predecessor 1 77123 .coefficient])

def exact77125RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77125RawTermsValid :
    exact77125RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77125 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42839⟩⟩) exact77125RawTerms .large 77124 .exactZero (none)

def event77126 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44369⟩⟩) 0 ⟨42839⟩ 77125

def event77127 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44369⟩⟩) 1 ⟨44368⟩ 77110

def event77128 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44369⟩⟩) (.sum [.predecessor 0 77126 .coefficient, .predecessor 1 77127 .coefficient])

def exact77129RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], [⟨.program ⟨257⟩, ⟨43825⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77129RawTermsValid :
    exact77129RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77129 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44369⟩⟩) exact77129RawTerms .large 77128 .exactZero (none)

def event77130 : Event := .preFoldPolynomial 77129 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], [⟨.program ⟨257⟩, ⟨43825⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact77131RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], [⟨.program ⟨257⟩, ⟨43825⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event77131 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨44369⟩⟩) 77130 exact77131RawTerms .large 77128 .exactZero (none)

def event77132 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨42620⟩⟩) ⟨⟨73⟩, ⟨52⟩, ⟨135⟩⟩ ⟨76966, 77132⟩

def event77133 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨43292⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43289⟩⟩]⟩) (1) 0 2 (.universal 77132 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43289⟩⟩]⟩) (none) 77131)

def event77134 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43292⟩⟩, .relation 77133 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩)

def event77135 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43292⟩⟩, .relation 77133 1, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩]⟩, (-1)⟩)

def event77136 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43292⟩⟩, .relation 77133 2, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], [⟨.program ⟨257⟩, ⟨43825⟩⟩]⟩, (1)⟩)

def event77137 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43292⟩⟩, .relation 77133 3, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact77138RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], [⟨.program ⟨257⟩, ⟨43825⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77138RawTermsValid :
    exact77138RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77138 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43292⟩⟩) exact77138RawTerms .large 76962 (.finite 202072841853861888) (some (76964))

def event77139 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44367⟩⟩) 0 ⟨43292⟩ 77138

def event77140 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44367⟩⟩) 1 ⟨44366⟩ 76952

def event77141 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44367⟩⟩) (.sum [.predecessor 0 77139 .coefficient, .predecessor 1 77140 .coefficient])

def event77142 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44367⟩⟩, .operator (⟨77138, 2⟩, ⟨76952, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], [⟨.program ⟨257⟩, ⟨43825⟩⟩]⟩, (-1)⟩)

def event77143 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44367⟩⟩, .operator (⟨77138, 1⟩, ⟨76952, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩]⟩, (1)⟩)

def event77144 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44367⟩⟩) (.sum [.result 77138 .summary, .result 76952 .summary])

def exact77145RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77145RawTermsValid :
    exact77145RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77145 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44367⟩⟩) exact77145RawTerms .large 77141 (.finite 2998273677530297008128) (some (77144))

def event77146 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44821⟩⟩) 0 ⟨44367⟩ 77145

def event77147 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44821⟩⟩) 1 ⟨44819⟩ 76868

def event77148 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44821⟩⟩) (.product (.predecessor 0 77146 .coefficient) (.predecessor 1 77147 .coefficient) (⟨false, false, none, none, none⟩))

def event77149 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44821⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩) [⟨.result 76868 .coefficient, false, none⟩])

def event77150 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44821⟩⟩) (.product (.result 77145 .summary) (.transfer 77149) (⟨false, false, none, none, none⟩))

def event77151 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44821⟩⟩, .operator (⟨77145, 0⟩, ⟨76868, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩, (1)⟩)

def event77152 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44821⟩⟩, .operator (⟨77145, 1⟩, ⟨76868, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩, (-1)⟩)

def event77153 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44821⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44819⟩⟩) ⟨43995⟩ 76865)

def event77154 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44821⟩⟩, .relation 77153 0, ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨43995⟩⟩]⟩, (-1)⟩)

def exact77155RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨43995⟩⟩]⟩, (-1)⟩]

theorem exact77155RawTermsValid :
    exact77155RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77155 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44821⟩⟩) exact77155RawTerms .large 77148 (.finite 32193718473625689247691015454720) (some (77150))

def event77156 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43656⟩⟩) 0 ⟨42837⟩ 3149

def event77157 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43656⟩⟩) (.authority (.relationPreimageSource ⟨90⟩))

def exact77158RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43656⟩⟩]⟩, (1)⟩]

theorem exact77158RawTermsValid :
    exact77158RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77158 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43656⟩⟩) exact77158RawTerms (.finite 5647228698) 77157 .exactZero (none)

def event77159 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43658⟩⟩) 0 ⟨43656⟩ 77158

def event77160 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43658⟩⟩) 1 ⟨2370⟩ 4

def event77161 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43658⟩⟩) (.scale (.predecessor 0 77159 .coefficient) (.value (.predecessor 1 77160 .coefficient)))

def exact77162RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43656⟩⟩]⟩, (1)⟩]

theorem exact77162RawTermsValid :
    exact77162RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77162 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43658⟩⟩) exact77162RawTerms (.finite 5647228698) 77161 .exactZero (none)

def event77163 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43659⟩⟩) 0 ⟨10368⟩ 75995

def event77164 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43659⟩⟩) 1 ⟨43658⟩ 77162

def event77165 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43659⟩⟩) (.product (.predecessor 0 77163 .coefficient) (.predecessor 1 77164 .coefficient) (⟨false, false, none, none, none⟩))

def event77166 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43659⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨43656⟩⟩]⟩) [⟨.result 77158 .coefficient, false, none⟩])

def event77167 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43659⟩⟩) (.product (.result 75995 .summary) (.transfer 77166) (⟨false, false, none, none, none⟩))

def event77168 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43659⟩⟩, .operator (⟨75995, 0⟩, ⟨77162, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43656⟩⟩]⟩, (1)⟩)

def event77169 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨43657⟩⟩)

def event77170 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event77171 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event77172 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event77173 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event77174 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event77175 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event77176 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event77177 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event77178 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 77177

def event77179 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 77175

def event77180 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 77178 .coefficient) (.value (.predecessor 1 77179 .coefficient)))

def event77181 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event77182 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 77181

def event77183 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 77173

def event77184 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 77182 .coefficient, .predecessor 1 77183 .coefficient])

def event77185 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event77186 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 77185

def event77187 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 77171

def event77188 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 77187 .coefficient))

def event77189 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event77190 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42618⟩⟩) 0 ⟨10325⟩ 77189

def event77191 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42618⟩⟩) (.authority (.programFamilyFact))

def exact77192RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42618⟩⟩], []⟩, (1)⟩]

theorem exact77192RawTermsValid :
    exact77192RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77192 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42618⟩⟩) exact77192RawTerms (.finite 52) 77191 .exactZero (none)

def event77193 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14571⟩⟩) 0 ⟨10325⟩ 77189

def event77194 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14571⟩⟩) (.authority (.programFamilyFact))

def exact77195RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩], []⟩, (1)⟩]

theorem exact77195RawTermsValid :
    exact77195RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77195 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14571⟩⟩) exact77195RawTerms (.finite 52) 77194 .exactZero (none)

def event77196 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42619⟩⟩) 0 ⟨14571⟩ 77195

def event77197 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42619⟩⟩) 1 ⟨42618⟩ 77192

def event77198 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42619⟩⟩) (.product (.predecessor 0 77196 .coefficient) (.predecessor 1 77197 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event77199 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42619⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], []⟩) [⟨.result 77195 .coefficient, true, some 1⟩, ⟨.result 77192 .coefficient, true, some 1⟩])

def event77200 : Event := .survivorFold (1) 77199

def exact77201RawTerms : List Term := []

theorem exact77201RawTermsValid :
    exact77201RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77201 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42619⟩⟩) exact77201RawTerms (.finite 2704) 77198 (.finite 2704) (some (77199))

def event77202 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42620⟩⟩) 0 ⟨42619⟩ 77201

def event77203 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42620⟩⟩) (.identity (.predecessor 0 77202 .coefficient))

def event77204 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42620⟩⟩) (.finite 2704)

def event77205 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42836⟩⟩) 0 ⟨42620⟩ 77204

def event77206 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42836⟩⟩) (.authority (.programFamilyFact))

def exact77207RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], []⟩, (1)⟩]

theorem exact77207RawTermsValid :
    exact77207RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77207 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42836⟩⟩) exact77207RawTerms (.finite 52) 77206 .exactZero (none)

def event77208 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42837⟩⟩) 0 ⟨42836⟩ 77207

def event77209 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42837⟩⟩) (.identity (.predecessor 0 77208 .coefficient))

def event77210 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42837⟩⟩) (.finite 52)

def event77211 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43656⟩⟩) 0 ⟨42837⟩ 77210

def event77212 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43656⟩⟩) (.authority (.relationPreimageSource ⟨90⟩))

def exact77213RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43656⟩⟩]⟩, (1)⟩]

theorem exact77213RawTermsValid :
    exact77213RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77213 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43656⟩⟩) exact77213RawTerms (.finite 5647228698) 77212 .exactZero (none)

def event77214 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact77215RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact77215RawTermsValid :
    exact77215RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77215 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact77215RawTerms .large 77214 .exactZero (none)

def event77216 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43657⟩⟩) 0 ⟨35⟩ 77215

def event77217 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43657⟩⟩) 1 ⟨43656⟩ 77213

def event77218 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43657⟩⟩) (.product (.predecessor 0 77216 .coefficient) (.predecessor 1 77217 .coefficient) (⟨false, false, none, none, none⟩))

def event77219 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43657⟩⟩, .operator (⟨77215, 0⟩, ⟨77213, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43656⟩⟩]⟩, (1)⟩)

def exact77220RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43656⟩⟩]⟩, (1)⟩]

theorem exact77220RawTermsValid :
    exact77220RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77220 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43657⟩⟩) exact77220RawTerms .large 77218 .exactZero (none)

def event77221 : Event := .preFoldPolynomial 77220 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43656⟩⟩]⟩, (1)⟩] .exactZero none

def exact77222RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43656⟩⟩]⟩, (1)⟩]

def event77222 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨43657⟩⟩) 77221 exact77222RawTerms .large 77218 .exactZero (none)

def event77223 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨44823⟩⟩)

def event77224 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event77225 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event77226 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.authority (.operator))

def event77227 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10267⟩⟩) (.finite 15)

def event77228 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event77229 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event77230 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event77231 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event77232 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 77231

def event77233 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 77229

def event77234 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 77232 .coefficient) (.value (.predecessor 1 77233 .coefficient)))

def event77235 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event77236 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 0 ⟨392⟩ 77235

def event77237 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10269⟩⟩) 1 ⟨10267⟩ 77227

def event77238 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.sum [.predecessor 0 77236 .coefficient, .predecessor 1 77237 .coefficient])

def event77239 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10269⟩⟩) (.finite 655355)

def event77240 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 0 ⟨10269⟩ 77239

def event77241 : Event := .predecessor (⟨.program ⟨257⟩, ⟨10325⟩⟩) 1 ⟨5426⟩ 77225

def event77242 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.identity (.predecessor 1 77241 .coefficient))

def event77243 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨10325⟩⟩) (.finite 655360)

def event77244 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42618⟩⟩) 0 ⟨10325⟩ 77243

def event77245 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42618⟩⟩) (.authority (.programFamilyFact))

def exact77246RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42618⟩⟩], []⟩, (1)⟩]

theorem exact77246RawTermsValid :
    exact77246RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77246 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42618⟩⟩) exact77246RawTerms (.finite 52) 77245 .exactZero (none)

def event77247 : Event := .predecessor (⟨.program ⟨257⟩, ⟨14571⟩⟩) 0 ⟨10325⟩ 77243

def event77248 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨14571⟩⟩) (.authority (.programFamilyFact))

def exact77249RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩], []⟩, (1)⟩]

theorem exact77249RawTermsValid :
    exact77249RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77249 : Event := .resultExact (⟨.program ⟨257⟩, ⟨14571⟩⟩) exact77249RawTerms (.finite 52) 77248 .exactZero (none)

def event77250 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42619⟩⟩) 0 ⟨14571⟩ 77249

def event77251 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42619⟩⟩) 1 ⟨42618⟩ 77246

def event77252 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42619⟩⟩) (.product (.predecessor 0 77250 .coefficient) (.predecessor 1 77251 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event77253 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨42619⟩⟩, .operator (⟨77249, 0⟩, ⟨77246, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], []⟩, (1)⟩)

def exact77254RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], []⟩, (1)⟩]

theorem exact77254RawTermsValid :
    exact77254RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77254 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42619⟩⟩) exact77254RawTerms (.finite 2704) 77252 .exactZero (none)

def event77255 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42620⟩⟩) 0 ⟨42619⟩ 77254

def event77256 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42620⟩⟩) (.identity (.predecessor 0 77255 .coefficient))

def event77257 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42620⟩⟩) (.finite 2704)

def event77258 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42836⟩⟩) 0 ⟨42620⟩ 77257

def event77259 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42836⟩⟩) (.authority (.programFamilyFact))

def exact77260RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], []⟩, (1)⟩]

theorem exact77260RawTermsValid :
    exact77260RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77260 : Event := .resultExact (⟨.program ⟨257⟩, ⟨42836⟩⟩) exact77260RawTerms (.finite 52) 77259 .exactZero (none)

def event77261 : Event := .predecessor (⟨.program ⟨257⟩, ⟨42837⟩⟩) 0 ⟨42836⟩ 77260

def event77262 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨42837⟩⟩) (.identity (.predecessor 0 77261 .coefficient))

def event77263 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨42837⟩⟩) (.finite 52)

def event77264 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43993⟩⟩) 0 ⟨42837⟩ 77263

def event77265 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43993⟩⟩) (.authority (.programFamilyFact))

def event77266 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨43993⟩⟩) (.finite 3720)

def event77267 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event77268 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43995⟩⟩) 0 ⟨7177⟩ 77267

def event77269 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43995⟩⟩) 1 ⟨43993⟩ 77266

def event77270 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43995⟩⟩) (.authority (.operator))

def exact77271RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨43995⟩⟩]⟩, (1)⟩]

theorem exact77271RawTermsValid :
    exact77271RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77271 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43995⟩⟩) exact77271RawTerms .large 77270 .exactZero (none)

def event77272 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44819⟩⟩) 0 ⟨43995⟩ 77271

def event77273 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44819⟩⟩) (.authority (.operator))

def exact77274RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩, (1)⟩]

theorem exact77274RawTermsValid :
    exact77274RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77274 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44819⟩⟩) exact77274RawTerms (.finite 8192) 77273 .exactZero (none)

def event77275 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event77276 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event77277 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44170⟩⟩) 0 ⟨42837⟩ 77263

def event77278 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44170⟩⟩) 1 ⟨136⟩ 77276

def event77279 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44170⟩⟩) (.sum [.predecessor 0 77277 .coefficient, .predecessor 1 77278 .coefficient])

def event77280 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨44170⟩⟩) (.finite 52)

def event77281 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44171⟩⟩) 0 ⟨44170⟩ 77280

def event77282 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44171⟩⟩) (.identity (.predecessor 0 77281 .coefficient))

def exact77283RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], []⟩, (1)⟩]

theorem exact77283RawTermsValid :
    exact77283RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77283 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44171⟩⟩) exact77283RawTerms (.finite 52) 77282 .exactZero (none)

def event77284 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact77285RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact77285RawTermsValid :
    exact77285RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77285 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact77285RawTerms .large 77284 .exactZero (none)

def event77286 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44172⟩⟩) 0 ⟨6908⟩ 77285

def event77287 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44172⟩⟩) 1 ⟨44171⟩ 77283

def event77288 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44172⟩⟩) (.product (.predecessor 0 77286 .coefficient) (.predecessor 1 77287 .coefficient) (⟨false, false, none, none, none⟩))

def event77289 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44172⟩⟩, .operator (⟨77285, 0⟩, ⟨77283, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact77290RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact77290RawTermsValid :
    exact77290RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77290 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44172⟩⟩) exact77290RawTerms .large 77288 .exactZero (none)

def event77291 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7194⟩⟩) 0 ⟨7177⟩ 77267

def event77292 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7194⟩⟩) (.authority (.operator))

def exact77293RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩]

theorem exact77293RawTermsValid :
    exact77293RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77293 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7194⟩⟩) exact77293RawTerms .large 77292 .exactZero (none)

def event77294 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44173⟩⟩) 0 ⟨7194⟩ 77293

def event77295 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44173⟩⟩) 1 ⟨44172⟩ 77290

def event77296 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44173⟩⟩) (.sum [.predecessor 0 77294 .coefficient, .predecessor 1 77295 .coefficient])

def exact77297RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact77297RawTermsValid :
    exact77297RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77297 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44173⟩⟩) exact77297RawTerms .large 77296 .exactZero (none)

def event77298 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44820⟩⟩) 0 ⟨44173⟩ 77297

def event77299 : Event := .predecessor (⟨.program ⟨257⟩, ⟨44820⟩⟩) 1 ⟨44819⟩ 77274

def event77300 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨44820⟩⟩) (.product (.predecessor 0 77298 .coefficient) (.predecessor 1 77299 .coefficient) (⟨false, false, none, none, none⟩))

def event77301 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44820⟩⟩, .operator (⟨77297, 0⟩, ⟨77274, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩, (1)⟩)

def event77302 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44820⟩⟩, .operator (⟨77297, 1⟩, ⟨77274, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩, (-1)⟩)

def event77303 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨44820⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44819⟩⟩) ⟨43995⟩ 77271)

def event77304 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨44820⟩⟩, .relation 77303 0, ⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨43995⟩⟩]⟩, (-1)⟩)

def exact77305RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨43995⟩⟩]⟩, (-1)⟩]

theorem exact77305RawTermsValid :
    exact77305RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77305 : Event := .resultExact (⟨.program ⟨257⟩, ⟨44820⟩⟩) exact77305RawTerms .large 77300 .exactZero (none)

def event77306 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43077⟩⟩) 0 ⟨42837⟩ 77263

def event77307 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43077⟩⟩) (.authority (.programFamilyFact))

def exact77308RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨43077⟩⟩], []⟩, (1)⟩]

theorem exact77308RawTermsValid :
    exact77308RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event77308 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43077⟩⟩) exact77308RawTerms (.finite 63) 77307 .exactZero (none)

def event77309 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43078⟩⟩) 0 ⟨6908⟩ 77285

def event77310 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43078⟩⟩) 1 ⟨43077⟩ 77308

def event77311 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43078⟩⟩) (.product (.predecessor 0 77309 .coefficient) (.predecessor 1 77310 .coefficient) (⟨false, true, none, none, some 1⟩))

def eventLeaf4816 : Array AnnotatedEvent := #[
  { event := event77056
    frameStart := 77014 },
  { event := event77057
    frameStart := 77014 },
  { event := event77058
    frameStart := 77014 },
  { event := event77059
    frameStart := 77014 },
  { event := event77060
    frameStart := 77014 },
  { event := event77061
    frameStart := 77014 },
  { event := event77062
    frameStart := 77014 },
  { event := event77063
    frameStart := 77014 },
  { event := event77064
    frameStart := 77014 },
  { event := event77065
    frameStart := 77014 },
  { event := event77066
    frameStart := 77014 },
  { event := event77067
    frameStart := 77014 },
  { event := event77068
    frameStart := 77014 },
  { event := event77069
    frameStart := 77014 },
  { event := event77070
    frameStart := 77014 },
  { event := event77071
    frameStart := 77014 }
]

def eventLeaf4817 : Array AnnotatedEvent := #[
  { event := event77072
    frameStart := 77014 },
  { event := event77073
    frameStart := 77014 },
  { event := event77074
    frameStart := 77014 },
  { event := event77075
    frameStart := 77014 },
  { event := event77076
    frameStart := 77014 },
  { event := event77077
    frameStart := 77014 },
  { event := event77078
    frameStart := 77014 },
  { event := event77079
    frameStart := 77014 },
  { event := event77080
    frameStart := 77014 },
  { event := event77081
    frameStart := 77014 },
  { event := event77082
    frameStart := 77014 },
  { event := event77083
    frameStart := 77014 },
  { event := event77084
    frameStart := 77014 },
  { event := event77085
    frameStart := 77014 },
  { event := event77086
    frameStart := 77014 },
  { event := event77087
    frameStart := 77014 }
]

def eventLeaf4818 : Array AnnotatedEvent := #[
  { event := event77088
    frameStart := 77014 },
  { event := event77089
    frameStart := 77014 },
  { event := event77090
    frameStart := 77014 },
  { event := event77091
    frameStart := 77014 },
  { event := event77092
    frameStart := 77014 },
  { event := event77093
    frameStart := 77014 },
  { event := event77094
    frameStart := 77014 },
  { event := event77095
    frameStart := 77014 },
  { event := event77096
    frameStart := 77014 },
  { event := event77097
    frameStart := 77014 },
  { event := event77098
    frameStart := 77014 },
  { event := event77099
    frameStart := 77014 },
  { event := event77100
    frameStart := 77014 },
  { event := event77101
    frameStart := 77014 },
  { event := event77102
    frameStart := 77014 },
  { event := event77103
    frameStart := 77014 }
]

def eventLeaf4819 : Array AnnotatedEvent := #[
  { event := event77104
    frameStart := 77014 },
  { event := event77105
    frameStart := 77014 },
  { event := event77106
    frameStart := 77014 },
  { event := event77107
    frameStart := 77014 },
  { event := event77108
    frameStart := 77014 },
  { event := event77109
    frameStart := 77014 },
  { event := event77110
    frameStart := 77014 },
  { event := event77111
    frameStart := 77014 },
  { event := event77112
    frameStart := 77014 },
  { event := event77113
    frameStart := 77014 },
  { event := event77114
    frameStart := 77014 },
  { event := event77115
    frameStart := 77014 },
  { event := event77116
    frameStart := 77014 },
  { event := event77117
    frameStart := 77014 },
  { event := event77118
    frameStart := 77014 },
  { event := event77119
    frameStart := 77014 }
]

def eventLeaf4820 : Array AnnotatedEvent := #[
  { event := event77120
    frameStart := 77014 },
  { event := event77121
    frameStart := 77014 },
  { event := event77122
    frameStart := 77014 },
  { event := event77123
    frameStart := 77014 },
  { event := event77124
    frameStart := 77014 },
  { event := event77125
    frameStart := 77014 },
  { event := event77126
    frameStart := 77014 },
  { event := event77127
    frameStart := 77014 },
  { event := event77128
    frameStart := 77014 },
  { event := event77129
    frameStart := 77014 },
  { event := event77130
    frameStart := 77014 },
  { event := event77131
    frameStart := 77014 },
  { event := event77132
    frameStart := 0 },
  { event := event77133
    frameStart := 0 },
  { event := event77134
    frameStart := 0 },
  { event := event77135
    frameStart := 0 }
]

def eventLeaf4821 : Array AnnotatedEvent := #[
  { event := event77136
    frameStart := 0 },
  { event := event77137
    frameStart := 0 },
  { event := event77138
    frameStart := 0 },
  { event := event77139
    frameStart := 0 },
  { event := event77140
    frameStart := 0 },
  { event := event77141
    frameStart := 0 },
  { event := event77142
    frameStart := 0 },
  { event := event77143
    frameStart := 0 },
  { event := event77144
    frameStart := 0 },
  { event := event77145
    frameStart := 0 },
  { event := event77146
    frameStart := 0 },
  { event := event77147
    frameStart := 0 },
  { event := event77148
    frameStart := 0 },
  { event := event77149
    frameStart := 0 },
  { event := event77150
    frameStart := 0 },
  { event := event77151
    frameStart := 0 }
]

def eventLeaf4822 : Array AnnotatedEvent := #[
  { event := event77152
    frameStart := 0 },
  { event := event77153
    frameStart := 0 },
  { event := event77154
    frameStart := 0 },
  { event := event77155
    frameStart := 0 },
  { event := event77156
    frameStart := 0 },
  { event := event77157
    frameStart := 0 },
  { event := event77158
    frameStart := 0 },
  { event := event77159
    frameStart := 0 },
  { event := event77160
    frameStart := 0 },
  { event := event77161
    frameStart := 0 },
  { event := event77162
    frameStart := 0 },
  { event := event77163
    frameStart := 0 },
  { event := event77164
    frameStart := 0 },
  { event := event77165
    frameStart := 0 },
  { event := event77166
    frameStart := 0 },
  { event := event77167
    frameStart := 0 }
]

def eventLeaf4823 : Array AnnotatedEvent := #[
  { event := event77168
    frameStart := 0 },
  { event := event77169
    frameStart := 77169 },
  { event := event77170
    frameStart := 77169 },
  { event := event77171
    frameStart := 77169 },
  { event := event77172
    frameStart := 77169 },
  { event := event77173
    frameStart := 77169 },
  { event := event77174
    frameStart := 77169 },
  { event := event77175
    frameStart := 77169 },
  { event := event77176
    frameStart := 77169 },
  { event := event77177
    frameStart := 77169 },
  { event := event77178
    frameStart := 77169 },
  { event := event77179
    frameStart := 77169 },
  { event := event77180
    frameStart := 77169 },
  { event := event77181
    frameStart := 77169 },
  { event := event77182
    frameStart := 77169 },
  { event := event77183
    frameStart := 77169 }
]

def eventLeaf4824 : Array AnnotatedEvent := #[
  { event := event77184
    frameStart := 77169 },
  { event := event77185
    frameStart := 77169 },
  { event := event77186
    frameStart := 77169 },
  { event := event77187
    frameStart := 77169 },
  { event := event77188
    frameStart := 77169 },
  { event := event77189
    frameStart := 77169 },
  { event := event77190
    frameStart := 77169 },
  { event := event77191
    frameStart := 77169 },
  { event := event77192
    frameStart := 77169 },
  { event := event77193
    frameStart := 77169 },
  { event := event77194
    frameStart := 77169 },
  { event := event77195
    frameStart := 77169 },
  { event := event77196
    frameStart := 77169 },
  { event := event77197
    frameStart := 77169 },
  { event := event77198
    frameStart := 77169 },
  { event := event77199
    frameStart := 77169 }
]

def eventLeaf4825 : Array AnnotatedEvent := #[
  { event := event77200
    frameStart := 77169 },
  { event := event77201
    frameStart := 77169 },
  { event := event77202
    frameStart := 77169 },
  { event := event77203
    frameStart := 77169 },
  { event := event77204
    frameStart := 77169 },
  { event := event77205
    frameStart := 77169 },
  { event := event77206
    frameStart := 77169 },
  { event := event77207
    frameStart := 77169 },
  { event := event77208
    frameStart := 77169 },
  { event := event77209
    frameStart := 77169 },
  { event := event77210
    frameStart := 77169 },
  { event := event77211
    frameStart := 77169 },
  { event := event77212
    frameStart := 77169 },
  { event := event77213
    frameStart := 77169 },
  { event := event77214
    frameStart := 77169 },
  { event := event77215
    frameStart := 77169 }
]

def eventLeaf4826 : Array AnnotatedEvent := #[
  { event := event77216
    frameStart := 77169 },
  { event := event77217
    frameStart := 77169 },
  { event := event77218
    frameStart := 77169 },
  { event := event77219
    frameStart := 77169 },
  { event := event77220
    frameStart := 77169 },
  { event := event77221
    frameStart := 77169 },
  { event := event77222
    frameStart := 77169 },
  { event := event77223
    frameStart := 77223 },
  { event := event77224
    frameStart := 77223 },
  { event := event77225
    frameStart := 77223 },
  { event := event77226
    frameStart := 77223 },
  { event := event77227
    frameStart := 77223 },
  { event := event77228
    frameStart := 77223 },
  { event := event77229
    frameStart := 77223 },
  { event := event77230
    frameStart := 77223 },
  { event := event77231
    frameStart := 77223 }
]

def eventLeaf4827 : Array AnnotatedEvent := #[
  { event := event77232
    frameStart := 77223 },
  { event := event77233
    frameStart := 77223 },
  { event := event77234
    frameStart := 77223 },
  { event := event77235
    frameStart := 77223 },
  { event := event77236
    frameStart := 77223 },
  { event := event77237
    frameStart := 77223 },
  { event := event77238
    frameStart := 77223 },
  { event := event77239
    frameStart := 77223 },
  { event := event77240
    frameStart := 77223 },
  { event := event77241
    frameStart := 77223 },
  { event := event77242
    frameStart := 77223 },
  { event := event77243
    frameStart := 77223 },
  { event := event77244
    frameStart := 77223 },
  { event := event77245
    frameStart := 77223 },
  { event := event77246
    frameStart := 77223 },
  { event := event77247
    frameStart := 77223 }
]

def eventLeaf4828 : Array AnnotatedEvent := #[
  { event := event77248
    frameStart := 77223 },
  { event := event77249
    frameStart := 77223 },
  { event := event77250
    frameStart := 77223 },
  { event := event77251
    frameStart := 77223 },
  { event := event77252
    frameStart := 77223 },
  { event := event77253
    frameStart := 77223 },
  { event := event77254
    frameStart := 77223 },
  { event := event77255
    frameStart := 77223 },
  { event := event77256
    frameStart := 77223 },
  { event := event77257
    frameStart := 77223 },
  { event := event77258
    frameStart := 77223 },
  { event := event77259
    frameStart := 77223 },
  { event := event77260
    frameStart := 77223 },
  { event := event77261
    frameStart := 77223 },
  { event := event77262
    frameStart := 77223 },
  { event := event77263
    frameStart := 77223 }
]

def eventLeaf4829 : Array AnnotatedEvent := #[
  { event := event77264
    frameStart := 77223 },
  { event := event77265
    frameStart := 77223 },
  { event := event77266
    frameStart := 77223 },
  { event := event77267
    frameStart := 77223 },
  { event := event77268
    frameStart := 77223 },
  { event := event77269
    frameStart := 77223 },
  { event := event77270
    frameStart := 77223 },
  { event := event77271
    frameStart := 77223 },
  { event := event77272
    frameStart := 77223 },
  { event := event77273
    frameStart := 77223 },
  { event := event77274
    frameStart := 77223 },
  { event := event77275
    frameStart := 77223 },
  { event := event77276
    frameStart := 77223 },
  { event := event77277
    frameStart := 77223 },
  { event := event77278
    frameStart := 77223 },
  { event := event77279
    frameStart := 77223 }
]

def eventLeaf4830 : Array AnnotatedEvent := #[
  { event := event77280
    frameStart := 77223 },
  { event := event77281
    frameStart := 77223 },
  { event := event77282
    frameStart := 77223 },
  { event := event77283
    frameStart := 77223 },
  { event := event77284
    frameStart := 77223 },
  { event := event77285
    frameStart := 77223 },
  { event := event77286
    frameStart := 77223 },
  { event := event77287
    frameStart := 77223 },
  { event := event77288
    frameStart := 77223 },
  { event := event77289
    frameStart := 77223 },
  { event := event77290
    frameStart := 77223 },
  { event := event77291
    frameStart := 77223 },
  { event := event77292
    frameStart := 77223 },
  { event := event77293
    frameStart := 77223 },
  { event := event77294
    frameStart := 77223 },
  { event := event77295
    frameStart := 77223 }
]

def eventLeaf4831 : Array AnnotatedEvent := #[
  { event := event77296
    frameStart := 77223 },
  { event := event77297
    frameStart := 77223 },
  { event := event77298
    frameStart := 77223 },
  { event := event77299
    frameStart := 77223 },
  { event := event77300
    frameStart := 77223 },
  { event := event77301
    frameStart := 77223 },
  { event := event77302
    frameStart := 77223 },
  { event := event77303
    frameStart := 77223 },
  { event := event77304
    frameStart := 77223 },
  { event := event77305
    frameStart := 77223 },
  { event := event77306
    frameStart := 77223 },
  { event := event77307
    frameStart := 77223 },
  { event := event77308
    frameStart := 77223 },
  { event := event77309
    frameStart := 77223 },
  { event := event77310
    frameStart := 77223 },
  { event := event77311
    frameStart := 77223 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events301
