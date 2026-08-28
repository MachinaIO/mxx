import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events344

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact88064RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26564⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨23784⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88064RawTermsValid :
    exact88064RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88064 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20539⟩⟩) exact88064RawTerms .large 87896 (.finite 1811303510016) (some (87898))

def event88065 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26567⟩⟩) 0 ⟨20539⟩ 88064

def event88066 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26567⟩⟩) 1 ⟨26566⟩ 87886

def event88067 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26567⟩⟩) (.sum [.predecessor 0 88065 .coefficient, .predecessor 1 88066 .coefficient])

def event88068 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26567⟩⟩, .operator (⟨88064, 0⟩, ⟨87886, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26564⟩⟩]⟩, (1)⟩)

def event88069 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26567⟩⟩, .operator (⟨88064, 2⟩, ⟨87886, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14953⟩⟩], [⟨.program ⟨214⟩, ⟨23784⟩⟩]⟩, (-1)⟩)

def event88070 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26567⟩⟩) (.sum [.result 88064 .summary, .result 87886 .summary])

def exact88071RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88071RawTermsValid :
    exact88071RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88071 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26567⟩⟩) exact88071RawTerms .large 88067 (.finite 1291900380601931935744) (some (88070))

def event88072 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23719⟩⟩) 0 ⟨14793⟩ 4236

def event88073 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23719⟩⟩) (.authority (.programFamilyFact))

def event88074 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23719⟩⟩) (.finite 3720)

def event88075 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23721⟩⟩) 0 ⟨6689⟩ 5477

def event88076 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23721⟩⟩) 1 ⟨23719⟩ 88074

def event88077 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23721⟩⟩) (.authority (.operator))

def exact88078RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23721⟩⟩]⟩, (1)⟩]

theorem exact88078RawTermsValid :
    exact88078RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88078 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23721⟩⟩) exact88078RawTerms .large 88077 .exactZero (none)

def event88079 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26358⟩⟩) 0 ⟨23721⟩ 88078

def event88080 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26358⟩⟩) (.authority (.operator))

def exact88081RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26358⟩⟩]⟩, (1)⟩]

theorem exact88081RawTermsValid :
    exact88081RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88081 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26358⟩⟩) exact88081RawTerms (.finite 8192) 88080 .exactZero (none)

def event88082 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22953⟩⟩) 0 ⟨10482⟩ 4230

def event88083 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22953⟩⟩) (.authority (.programFamilyFact))

def event88084 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨22953⟩⟩) (.finite 3720)

def event88085 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22954⟩⟩) 0 ⟨6689⟩ 5477

def event88086 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22954⟩⟩) 1 ⟨22953⟩ 88084

def event88087 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22954⟩⟩) (.authority (.operator))

def exact88088RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22954⟩⟩]⟩, (1)⟩]

theorem exact88088RawTermsValid :
    exact88088RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88088 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22954⟩⟩) exact88088RawTerms .large 88087 .exactZero (none)

def event88089 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24911⟩⟩) 0 ⟨22954⟩ 88088

def event88090 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24911⟩⟩) (.authority (.operator))

def exact88091RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24911⟩⟩]⟩, (1)⟩]

theorem exact88091RawTermsValid :
    exact88091RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88091 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24911⟩⟩) exact88091RawTerms (.finite 8192) 88090 .exactZero (none)

def event88092 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10483⟩⟩) 0 ⟨10480⟩ 4219

def event88093 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10483⟩⟩) 1 ⟨6567⟩ 79920

def event88094 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10483⟩⟩) (.tensor (.predecessor 0 88092 .coefficient) (.predecessor 1 88093 .coefficient) true false)

def event88095 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10483⟩⟩, .operator (⟨4219, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact88096RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact88096RawTermsValid :
    exact88096RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88096 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10483⟩⟩) exact88096RawTerms .large 88094 .exactZero (none)

def event88097 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7228⟩⟩) 0 ⟨5539⟩ 79790

def event88098 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7228⟩⟩) 1 ⟨6772⟩ 14989

def event88099 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7228⟩⟩) (.product (.predecessor 0 88097 .coefficient) (.predecessor 1 88098 .coefficient) (⟨false, false, none, none, none⟩))

def event88100 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7228⟩⟩, .operator (⟨79790, 0⟩, ⟨14989, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩)

def exact88101RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩]

theorem exact88101RawTermsValid :
    exact88101RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88101 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7228⟩⟩) exact88101RawTerms .large 88099 .exactZero (none)

def event88102 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10484⟩⟩) 0 ⟨7228⟩ 88101

def event88103 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10484⟩⟩) 1 ⟨10483⟩ 88096

def event88104 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10484⟩⟩) (.sum [.predecessor 0 88102 .coefficient, .predecessor 1 88103 .coefficient])

def exact88105RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88105RawTermsValid :
    exact88105RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88105 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10484⟩⟩) exact88105RawTerms .large 88104 .exactZero (none)

def event88106 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10485⟩⟩) 0 ⟨10484⟩ 88105

def event88107 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10485⟩⟩) 1 ⟨86⟩ 14981

def event88108 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10485⟩⟩) (.sum [.predecessor 0 88106 .coefficient, .predecessor 1 88107 .coefficient])

def event88109 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10485⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨86⟩⟩]⟩) [⟨.result 14981 .coefficient, false, none⟩])

def event88110 : Event := .survivorFold (1) 88109

def exact88111RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88111RawTermsValid :
    exact88111RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88111 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10485⟩⟩) exact88111RawTerms .large 88108 (.finite 26) (some (88109))

def event88112 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10486⟩⟩) 0 ⟨10485⟩ 88111

def event88113 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10486⟩⟩) 1 ⟨9400⟩ 4222

def event88114 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10486⟩⟩) (.product (.predecessor 0 88112 .coefficient) (.predecessor 1 88113 .coefficient) (⟨false, true, none, none, some 1⟩))

def event88115 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10486⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩], []⟩) [⟨.result 4222 .coefficient, true, some 1⟩])

def event88116 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10486⟩⟩) (.product (.result 88111 .summary) (.transfer 88115) (⟨false, false, none, none, none⟩))

def event88117 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10486⟩⟩, .operator (⟨88111, 1⟩, ⟨4222, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event88118 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10486⟩⟩, .operator (⟨88111, 0⟩, ⟨4222, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩)

def exact88119RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88119RawTermsValid :
    exact88119RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88119 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10486⟩⟩) exact88119RawTerms .large 88114 (.finite 1664) (some (88116))

def event88120 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9401⟩⟩) 0 ⟨9400⟩ 4222

def event88121 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9401⟩⟩) 1 ⟨6567⟩ 79920

def event88122 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9401⟩⟩) (.tensor (.predecessor 0 88120 .coefficient) (.predecessor 1 88121 .coefficient) true false)

def event88123 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9401⟩⟩, .operator (⟨4222, 0⟩, ⟨79920, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact88124RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact88124RawTermsValid :
    exact88124RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88124 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9401⟩⟩) exact88124RawTerms .large 88122 .exactZero (none)

def event88125 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7227⟩⟩) 0 ⟨5539⟩ 79790

def event88126 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7227⟩⟩) 1 ⟨6771⟩ 15030

def event88127 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7227⟩⟩) (.product (.predecessor 0 88125 .coefficient) (.predecessor 1 88126 .coefficient) (⟨false, false, none, none, none⟩))

def event88128 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7227⟩⟩, .operator (⟨79790, 0⟩, ⟨15030, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩]⟩, (1)⟩)

def exact88129RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩]⟩, (1)⟩]

theorem exact88129RawTermsValid :
    exact88129RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88129 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7227⟩⟩) exact88129RawTerms .large 88127 .exactZero (none)

def event88130 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9402⟩⟩) 0 ⟨7227⟩ 88129

def event88131 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9402⟩⟩) 1 ⟨9401⟩ 88124

def event88132 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9402⟩⟩) (.sum [.predecessor 0 88130 .coefficient, .predecessor 1 88131 .coefficient])

def exact88133RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88133RawTermsValid :
    exact88133RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88133 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9402⟩⟩) exact88133RawTerms .large 88132 .exactZero (none)

def event88134 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9403⟩⟩) 0 ⟨9402⟩ 88133

def event88135 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9403⟩⟩) 1 ⟨85⟩ 15022

def event88136 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9403⟩⟩) (.sum [.predecessor 0 88134 .coefficient, .predecessor 1 88135 .coefficient])

def event88137 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9403⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨85⟩⟩]⟩) [⟨.result 15022 .coefficient, false, none⟩])

def event88138 : Event := .survivorFold (1) 88137

def exact88139RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88139RawTermsValid :
    exact88139RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88139 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9403⟩⟩) exact88139RawTerms .large 88136 (.finite 26) (some (88137))

def event88140 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9404⟩⟩) 0 ⟨9403⟩ 88139

def event88141 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9404⟩⟩) 1 ⟨7832⟩ 15019

def event88142 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9404⟩⟩) (.product (.predecessor 0 88140 .coefficient) (.predecessor 1 88141 .coefficient) (⟨false, false, none, none, none⟩))

def event88143 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9404⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩) [⟨.result 15015 .coefficient, false, none⟩])

def event88144 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9404⟩⟩) (.product (.result 88139 .summary) (.transfer 88143) (⟨false, false, none, none, none⟩))

def event88145 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9404⟩⟩, .operator (⟨88139, 1⟩, ⟨15019, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩, (-1)⟩)

def event88146 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨9404⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7831⟩⟩) ⟨6772⟩ 14989)

def event88147 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9404⟩⟩, .relation 88146 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (-1)⟩)

def event88148 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9404⟩⟩, .operator (⟨88139, 0⟩, ⟨15019, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩, (1)⟩)

def exact88149RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (-1)⟩]

theorem exact88149RawTermsValid :
    exact88149RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88149 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9404⟩⟩) exact88149RawTerms .large 88142 (.finite 95420416) (some (88144))

def event88150 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10487⟩⟩) 0 ⟨9404⟩ 88149

def event88151 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10487⟩⟩) 1 ⟨10486⟩ 88119

def event88152 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10487⟩⟩) (.sum [.predecessor 0 88150 .coefficient, .predecessor 1 88151 .coefficient])

def event88153 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10487⟩⟩, .operator (⟨88149, 1⟩, ⟨88119, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩)

def event88154 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10487⟩⟩) (.sum [.result 88149 .summary, .result 88119 .summary])

def exact88155RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88155RawTermsValid :
    exact88155RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88155 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10487⟩⟩) exact88155RawTerms .large 88152 (.finite 95422080) (some (88154))

def event88156 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24912⟩⟩) 0 ⟨10487⟩ 88155

def event88157 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24912⟩⟩) 1 ⟨24911⟩ 88091

def event88158 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24912⟩⟩) (.product (.predecessor 0 88156 .coefficient) (.predecessor 1 88157 .coefficient) (⟨false, false, none, none, none⟩))

def event88159 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24912⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨24911⟩⟩]⟩) [⟨.result 88091 .coefficient, false, none⟩])

def event88160 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24912⟩⟩) (.product (.result 88155 .summary) (.transfer 88159) (⟨false, false, none, none, none⟩))

def event88161 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24912⟩⟩, .operator (⟨88155, 1⟩, ⟨88091, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24911⟩⟩]⟩, (-1)⟩)

def event88162 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨24912⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24911⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨24911⟩⟩) ⟨22954⟩ 88088)

def event88163 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24912⟩⟩, .relation 88162 0, ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨22954⟩⟩]⟩, (-1)⟩)

def event88164 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24912⟩⟩, .operator (⟨88155, 0⟩, ⟨88091, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24911⟩⟩]⟩, (1)⟩)

def exact88165RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24911⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨22954⟩⟩]⟩, (-1)⟩]

theorem exact88165RawTermsValid :
    exact88165RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88165 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24912⟩⟩) exact88165RawTerms .large 88158 (.finite 350200560353280) (some (88160))

def event88166 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19024⟩⟩) 0 ⟨10482⟩ 4230

def event88167 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19024⟩⟩) (.authority (.relationPreimageSource ⟨7⟩))

def exact88168RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19024⟩⟩]⟩, (1)⟩]

theorem exact88168RawTermsValid :
    exact88168RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88168 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19024⟩⟩) exact88168RawTerms (.finite 136065468) 88167 .exactZero (none)

def event88169 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19026⟩⟩) 0 ⟨19024⟩ 88168

def event88170 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19026⟩⟩) 1 ⟨2348⟩ 4

def event88171 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19026⟩⟩) (.scale (.predecessor 0 88169 .coefficient) (.value (.predecessor 1 88170 .coefficient)))

def exact88172RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19024⟩⟩]⟩, (1)⟩]

theorem exact88172RawTermsValid :
    exact88172RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88172 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19026⟩⟩) exact88172RawTerms (.finite 136065468) 88171 .exactZero (none)

def event88173 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19027⟩⟩) 0 ⟨5541⟩ 80012

def event88174 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19027⟩⟩) 1 ⟨19026⟩ 88172

def event88175 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19027⟩⟩) (.product (.predecessor 0 88173 .coefficient) (.predecessor 1 88174 .coefficient) (⟨false, false, none, none, none⟩))

def event88176 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19027⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨19024⟩⟩]⟩) [⟨.result 88168 .coefficient, false, none⟩])

def event88177 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19027⟩⟩) (.product (.result 80012 .summary) (.transfer 88176) (⟨false, false, none, none, none⟩))

def event88178 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19027⟩⟩, .operator (⟨80012, 0⟩, ⟨88172, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19024⟩⟩]⟩, (1)⟩)

def event88179 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨19025⟩⟩)

def event88180 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event88181 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event88182 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event88183 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event88184 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event88185 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event88186 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event88187 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event88188 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 88187

def event88189 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 88185

def event88190 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 88188 .coefficient) (.value (.predecessor 1 88189 .coefficient)))

def event88191 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event88192 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 88191

def event88193 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 88183

def event88194 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 88192 .coefficient, .predecessor 1 88193 .coefficient])

def event88195 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event88196 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 88195

def event88197 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 88181

def event88198 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 88197 .coefficient))

def event88199 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event88200 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10480⟩⟩) 0 ⟨5536⟩ 88199

def event88201 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10480⟩⟩) (.authority (.programFamilyFact))

def exact88202RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10480⟩⟩], []⟩, (1)⟩]

theorem exact88202RawTermsValid :
    exact88202RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88202 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10480⟩⟩) exact88202RawTerms (.finite 2) 88201 .exactZero (none)

def event88203 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9400⟩⟩) 0 ⟨5536⟩ 88199

def event88204 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9400⟩⟩) (.authority (.programFamilyFact))

def exact88205RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩], []⟩, (1)⟩]

theorem exact88205RawTermsValid :
    exact88205RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88205 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9400⟩⟩) exact88205RawTerms (.finite 2) 88204 .exactZero (none)

def event88206 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10481⟩⟩) 0 ⟨9400⟩ 88205

def event88207 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10481⟩⟩) 1 ⟨10480⟩ 88202

def event88208 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10481⟩⟩) (.product (.predecessor 0 88206 .coefficient) (.predecessor 1 88207 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event88209 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10481⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], []⟩) [⟨.result 88205 .coefficient, true, some 1⟩, ⟨.result 88202 .coefficient, true, some 1⟩])

def event88210 : Event := .survivorFold (1) 88209

def exact88211RawTerms : List Term := []

theorem exact88211RawTermsValid :
    exact88211RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88211 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10481⟩⟩) exact88211RawTerms (.finite 4) 88208 (.finite 4) (some (88209))

def event88212 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10482⟩⟩) 0 ⟨10481⟩ 88211

def event88213 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10482⟩⟩) (.identity (.predecessor 0 88212 .coefficient))

def event88214 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10482⟩⟩) (.finite 4)

def event88215 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19024⟩⟩) 0 ⟨10482⟩ 88214

def event88216 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19024⟩⟩) (.authority (.relationPreimageSource ⟨7⟩))

def exact88217RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19024⟩⟩]⟩, (1)⟩]

theorem exact88217RawTermsValid :
    exact88217RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88217 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19024⟩⟩) exact88217RawTerms (.finite 136065468) 88216 .exactZero (none)

def event88218 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact88219RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact88219RawTermsValid :
    exact88219RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88219 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact88219RawTerms .large 88218 .exactZero (none)

def event88220 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19025⟩⟩) 0 ⟨6⟩ 88219

def event88221 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19025⟩⟩) 1 ⟨19024⟩ 88217

def event88222 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19025⟩⟩) (.product (.predecessor 0 88220 .coefficient) (.predecessor 1 88221 .coefficient) (⟨false, false, none, none, none⟩))

def event88223 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19025⟩⟩, .operator (⟨88219, 0⟩, ⟨88217, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19024⟩⟩]⟩, (1)⟩)

def exact88224RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19024⟩⟩]⟩, (1)⟩]

theorem exact88224RawTermsValid :
    exact88224RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88224 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19025⟩⟩) exact88224RawTerms .large 88222 .exactZero (none)

def event88225 : Event := .preFoldPolynomial 88224 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19024⟩⟩]⟩, (1)⟩] .exactZero none

def exact88226RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19024⟩⟩]⟩, (1)⟩]

def event88226 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19025⟩⟩) 88225 exact88226RawTerms .large 88222 .exactZero (none)

def event88227 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨24915⟩⟩)

def event88228 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event88229 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event88230 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event88231 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event88232 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event88233 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event88234 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event88235 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event88236 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 88235

def event88237 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 88233

def event88238 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 88236 .coefficient) (.value (.predecessor 1 88237 .coefficient)))

def event88239 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event88240 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 0 ⟨5503⟩ 88239

def event88241 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5505⟩⟩) 1 ⟨2348⟩ 88231

def event88242 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.sum [.predecessor 0 88240 .coefficient, .predecessor 1 88241 .coefficient])

def event88243 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5505⟩⟩) (.finite 218)

def event88244 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 0 ⟨5505⟩ 88243

def event88245 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5536⟩⟩) 1 ⟨961⟩ 88229

def event88246 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.identity (.predecessor 1 88245 .coefficient))

def event88247 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5536⟩⟩) (.finite 224)

def event88248 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10480⟩⟩) 0 ⟨5536⟩ 88247

def event88249 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10480⟩⟩) (.authority (.programFamilyFact))

def exact88250RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10480⟩⟩], []⟩, (1)⟩]

theorem exact88250RawTermsValid :
    exact88250RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88250 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10480⟩⟩) exact88250RawTerms (.finite 2) 88249 .exactZero (none)

def event88251 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9400⟩⟩) 0 ⟨5536⟩ 88247

def event88252 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9400⟩⟩) (.authority (.programFamilyFact))

def exact88253RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩], []⟩, (1)⟩]

theorem exact88253RawTermsValid :
    exact88253RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88253 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9400⟩⟩) exact88253RawTerms (.finite 2) 88252 .exactZero (none)

def event88254 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10481⟩⟩) 0 ⟨9400⟩ 88253

def event88255 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10481⟩⟩) 1 ⟨10480⟩ 88250

def event88256 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10481⟩⟩) (.product (.predecessor 0 88254 .coefficient) (.predecessor 1 88255 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event88257 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10481⟩⟩, .operator (⟨88253, 0⟩, ⟨88250, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], []⟩, (1)⟩)

def exact88258RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], []⟩, (1)⟩]

theorem exact88258RawTermsValid :
    exact88258RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88258 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10481⟩⟩) exact88258RawTerms (.finite 4) 88256 .exactZero (none)

def event88259 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10482⟩⟩) 0 ⟨10481⟩ 88258

def event88260 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10482⟩⟩) (.identity (.predecessor 0 88259 .coefficient))

def event88261 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10482⟩⟩) (.finite 4)

def event88262 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22953⟩⟩) 0 ⟨10482⟩ 88261

def event88263 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22953⟩⟩) (.authority (.programFamilyFact))

def event88264 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨22953⟩⟩) (.finite 3720)

def event88265 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event88266 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22954⟩⟩) 0 ⟨6689⟩ 88265

def event88267 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22954⟩⟩) 1 ⟨22953⟩ 88264

def event88268 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22954⟩⟩) (.authority (.operator))

def exact88269RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22954⟩⟩]⟩, (1)⟩]

theorem exact88269RawTermsValid :
    exact88269RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88269 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22954⟩⟩) exact88269RawTerms .large 88268 .exactZero (none)

def event88270 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24911⟩⟩) 0 ⟨22954⟩ 88269

def event88271 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24911⟩⟩) (.authority (.operator))

def exact88272RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24911⟩⟩]⟩, (1)⟩]

theorem exact88272RawTermsValid :
    exact88272RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88272 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24911⟩⟩) exact88272RawTerms (.finite 8192) 88271 .exactZero (none)

def event88273 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event88274 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event88275 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10576⟩⟩) 0 ⟨10482⟩ 88261

def event88276 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10576⟩⟩) 1 ⟨110⟩ 88274

def event88277 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10576⟩⟩) (.sum [.predecessor 0 88275 .coefficient, .predecessor 1 88276 .coefficient])

def event88278 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10576⟩⟩) (.finite 4)

def event88279 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10577⟩⟩) 0 ⟨10576⟩ 88278

def event88280 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10577⟩⟩) (.identity (.predecessor 0 88279 .coefficient))

def exact88281RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], []⟩, (1)⟩]

theorem exact88281RawTermsValid :
    exact88281RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88281 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10577⟩⟩) exact88281RawTerms (.finite 4) 88280 .exactZero (none)

def event88282 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact88283RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact88283RawTermsValid :
    exact88283RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88283 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact88283RawTerms .large 88282 .exactZero (none)

def event88284 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10578⟩⟩) 0 ⟨6544⟩ 88283

def event88285 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10578⟩⟩) 1 ⟨10577⟩ 88281

def event88286 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10578⟩⟩) (.product (.predecessor 0 88284 .coefficient) (.predecessor 1 88285 .coefficient) (⟨false, false, none, none, none⟩))

def event88287 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10578⟩⟩, .operator (⟨88283, 0⟩, ⟨88281, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact88288RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact88288RawTermsValid :
    exact88288RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88288 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10578⟩⟩) exact88288RawTerms .large 88286 .exactZero (none)

def event88289 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 88265

def event88290 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact88291RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact88291RawTermsValid :
    exact88291RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88291 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact88291RawTerms .large 88290 .exactZero (none)

def event88292 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6772⟩⟩) 0 ⟨6757⟩ 88291

def event88293 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6772⟩⟩) (.identity (.predecessor 0 88292 .coefficient))

def exact88294RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩]

theorem exact88294RawTermsValid :
    exact88294RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88294 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6772⟩⟩) exact88294RawTerms .large 88293 .exactZero (none)

def event88295 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7831⟩⟩) 0 ⟨6772⟩ 88294

def event88296 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7831⟩⟩) (.authority (.operator))

def exact88297RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩, (1)⟩]

theorem exact88297RawTermsValid :
    exact88297RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88297 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7831⟩⟩) exact88297RawTerms (.finite 8192) 88296 .exactZero (none)

def event88298 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7832⟩⟩) 0 ⟨7831⟩ 88297

def event88299 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7832⟩⟩) 1 ⟨2348⟩ 88231

def event88300 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7832⟩⟩) (.scale (.predecessor 0 88298 .coefficient) (.value (.predecessor 1 88299 .coefficient)))

def exact88301RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩, (1)⟩]

theorem exact88301RawTermsValid :
    exact88301RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88301 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7832⟩⟩) exact88301RawTerms (.finite 8192) 88300 .exactZero (none)

def event88302 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6771⟩⟩) 0 ⟨6757⟩ 88291

def event88303 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6771⟩⟩) (.identity (.predecessor 0 88302 .coefficient))

def exact88304RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩]⟩, (1)⟩]

theorem exact88304RawTermsValid :
    exact88304RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88304 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6771⟩⟩) exact88304RawTerms .large 88303 .exactZero (none)

def event88305 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7833⟩⟩) 0 ⟨6771⟩ 88304

def event88306 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7833⟩⟩) 1 ⟨7832⟩ 88301

def event88307 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7833⟩⟩) (.product (.predecessor 0 88305 .coefficient) (.predecessor 1 88306 .coefficient) (⟨false, false, none, none, none⟩))

def event88308 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7833⟩⟩, .operator (⟨88304, 0⟩, ⟨88301, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩, (1)⟩)

def exact88309RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩, (1)⟩]

theorem exact88309RawTermsValid :
    exact88309RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88309 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7833⟩⟩) exact88309RawTerms .large 88307 .exactZero (none)

def event88310 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10579⟩⟩) 0 ⟨7833⟩ 88309

def event88311 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10579⟩⟩) 1 ⟨10578⟩ 88288

def event88312 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10579⟩⟩) (.sum [.predecessor 0 88310 .coefficient, .predecessor 1 88311 .coefficient])

def exact88313RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact88313RawTermsValid :
    exact88313RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event88313 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10579⟩⟩) exact88313RawTerms .large 88312 .exactZero (none)

def event88314 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24914⟩⟩) 0 ⟨10579⟩ 88313

def event88315 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24914⟩⟩) 1 ⟨24911⟩ 88272

def event88316 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24914⟩⟩) (.product (.predecessor 0 88314 .coefficient) (.predecessor 1 88315 .coefficient) (⟨false, false, none, none, none⟩))

def event88317 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24914⟩⟩, .operator (⟨88313, 0⟩, ⟨88272, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24911⟩⟩]⟩, (1)⟩)

def event88318 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨24914⟩⟩, .operator (⟨88313, 1⟩, ⟨88272, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24911⟩⟩]⟩, (-1)⟩)

def event88319 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨24914⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨9400⟩⟩, ⟨.program ⟨214⟩, ⟨10480⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24911⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨24911⟩⟩) ⟨22954⟩ 88269)

def eventLeaf5504 : Array AnnotatedEvent := #[
  { event := event88064
    frameStart := 0 },
  { event := event88065
    frameStart := 0 },
  { event := event88066
    frameStart := 0 },
  { event := event88067
    frameStart := 0 },
  { event := event88068
    frameStart := 0 },
  { event := event88069
    frameStart := 0 },
  { event := event88070
    frameStart := 0 },
  { event := event88071
    frameStart := 0 },
  { event := event88072
    frameStart := 0 },
  { event := event88073
    frameStart := 0 },
  { event := event88074
    frameStart := 0 },
  { event := event88075
    frameStart := 0 },
  { event := event88076
    frameStart := 0 },
  { event := event88077
    frameStart := 0 },
  { event := event88078
    frameStart := 0 },
  { event := event88079
    frameStart := 0 }
]

def eventLeaf5505 : Array AnnotatedEvent := #[
  { event := event88080
    frameStart := 0 },
  { event := event88081
    frameStart := 0 },
  { event := event88082
    frameStart := 0 },
  { event := event88083
    frameStart := 0 },
  { event := event88084
    frameStart := 0 },
  { event := event88085
    frameStart := 0 },
  { event := event88086
    frameStart := 0 },
  { event := event88087
    frameStart := 0 },
  { event := event88088
    frameStart := 0 },
  { event := event88089
    frameStart := 0 },
  { event := event88090
    frameStart := 0 },
  { event := event88091
    frameStart := 0 },
  { event := event88092
    frameStart := 0 },
  { event := event88093
    frameStart := 0 },
  { event := event88094
    frameStart := 0 },
  { event := event88095
    frameStart := 0 }
]

def eventLeaf5506 : Array AnnotatedEvent := #[
  { event := event88096
    frameStart := 0 },
  { event := event88097
    frameStart := 0 },
  { event := event88098
    frameStart := 0 },
  { event := event88099
    frameStart := 0 },
  { event := event88100
    frameStart := 0 },
  { event := event88101
    frameStart := 0 },
  { event := event88102
    frameStart := 0 },
  { event := event88103
    frameStart := 0 },
  { event := event88104
    frameStart := 0 },
  { event := event88105
    frameStart := 0 },
  { event := event88106
    frameStart := 0 },
  { event := event88107
    frameStart := 0 },
  { event := event88108
    frameStart := 0 },
  { event := event88109
    frameStart := 0 },
  { event := event88110
    frameStart := 0 },
  { event := event88111
    frameStart := 0 }
]

def eventLeaf5507 : Array AnnotatedEvent := #[
  { event := event88112
    frameStart := 0 },
  { event := event88113
    frameStart := 0 },
  { event := event88114
    frameStart := 0 },
  { event := event88115
    frameStart := 0 },
  { event := event88116
    frameStart := 0 },
  { event := event88117
    frameStart := 0 },
  { event := event88118
    frameStart := 0 },
  { event := event88119
    frameStart := 0 },
  { event := event88120
    frameStart := 0 },
  { event := event88121
    frameStart := 0 },
  { event := event88122
    frameStart := 0 },
  { event := event88123
    frameStart := 0 },
  { event := event88124
    frameStart := 0 },
  { event := event88125
    frameStart := 0 },
  { event := event88126
    frameStart := 0 },
  { event := event88127
    frameStart := 0 }
]

def eventLeaf5508 : Array AnnotatedEvent := #[
  { event := event88128
    frameStart := 0 },
  { event := event88129
    frameStart := 0 },
  { event := event88130
    frameStart := 0 },
  { event := event88131
    frameStart := 0 },
  { event := event88132
    frameStart := 0 },
  { event := event88133
    frameStart := 0 },
  { event := event88134
    frameStart := 0 },
  { event := event88135
    frameStart := 0 },
  { event := event88136
    frameStart := 0 },
  { event := event88137
    frameStart := 0 },
  { event := event88138
    frameStart := 0 },
  { event := event88139
    frameStart := 0 },
  { event := event88140
    frameStart := 0 },
  { event := event88141
    frameStart := 0 },
  { event := event88142
    frameStart := 0 },
  { event := event88143
    frameStart := 0 }
]

def eventLeaf5509 : Array AnnotatedEvent := #[
  { event := event88144
    frameStart := 0 },
  { event := event88145
    frameStart := 0 },
  { event := event88146
    frameStart := 0 },
  { event := event88147
    frameStart := 0 },
  { event := event88148
    frameStart := 0 },
  { event := event88149
    frameStart := 0 },
  { event := event88150
    frameStart := 0 },
  { event := event88151
    frameStart := 0 },
  { event := event88152
    frameStart := 0 },
  { event := event88153
    frameStart := 0 },
  { event := event88154
    frameStart := 0 },
  { event := event88155
    frameStart := 0 },
  { event := event88156
    frameStart := 0 },
  { event := event88157
    frameStart := 0 },
  { event := event88158
    frameStart := 0 },
  { event := event88159
    frameStart := 0 }
]

def eventLeaf5510 : Array AnnotatedEvent := #[
  { event := event88160
    frameStart := 0 },
  { event := event88161
    frameStart := 0 },
  { event := event88162
    frameStart := 0 },
  { event := event88163
    frameStart := 0 },
  { event := event88164
    frameStart := 0 },
  { event := event88165
    frameStart := 0 },
  { event := event88166
    frameStart := 0 },
  { event := event88167
    frameStart := 0 },
  { event := event88168
    frameStart := 0 },
  { event := event88169
    frameStart := 0 },
  { event := event88170
    frameStart := 0 },
  { event := event88171
    frameStart := 0 },
  { event := event88172
    frameStart := 0 },
  { event := event88173
    frameStart := 0 },
  { event := event88174
    frameStart := 0 },
  { event := event88175
    frameStart := 0 }
]

def eventLeaf5511 : Array AnnotatedEvent := #[
  { event := event88176
    frameStart := 0 },
  { event := event88177
    frameStart := 0 },
  { event := event88178
    frameStart := 0 },
  { event := event88179
    frameStart := 88179 },
  { event := event88180
    frameStart := 88179 },
  { event := event88181
    frameStart := 88179 },
  { event := event88182
    frameStart := 88179 },
  { event := event88183
    frameStart := 88179 },
  { event := event88184
    frameStart := 88179 },
  { event := event88185
    frameStart := 88179 },
  { event := event88186
    frameStart := 88179 },
  { event := event88187
    frameStart := 88179 },
  { event := event88188
    frameStart := 88179 },
  { event := event88189
    frameStart := 88179 },
  { event := event88190
    frameStart := 88179 },
  { event := event88191
    frameStart := 88179 }
]

def eventLeaf5512 : Array AnnotatedEvent := #[
  { event := event88192
    frameStart := 88179 },
  { event := event88193
    frameStart := 88179 },
  { event := event88194
    frameStart := 88179 },
  { event := event88195
    frameStart := 88179 },
  { event := event88196
    frameStart := 88179 },
  { event := event88197
    frameStart := 88179 },
  { event := event88198
    frameStart := 88179 },
  { event := event88199
    frameStart := 88179 },
  { event := event88200
    frameStart := 88179 },
  { event := event88201
    frameStart := 88179 },
  { event := event88202
    frameStart := 88179 },
  { event := event88203
    frameStart := 88179 },
  { event := event88204
    frameStart := 88179 },
  { event := event88205
    frameStart := 88179 },
  { event := event88206
    frameStart := 88179 },
  { event := event88207
    frameStart := 88179 }
]

def eventLeaf5513 : Array AnnotatedEvent := #[
  { event := event88208
    frameStart := 88179 },
  { event := event88209
    frameStart := 88179 },
  { event := event88210
    frameStart := 88179 },
  { event := event88211
    frameStart := 88179 },
  { event := event88212
    frameStart := 88179 },
  { event := event88213
    frameStart := 88179 },
  { event := event88214
    frameStart := 88179 },
  { event := event88215
    frameStart := 88179 },
  { event := event88216
    frameStart := 88179 },
  { event := event88217
    frameStart := 88179 },
  { event := event88218
    frameStart := 88179 },
  { event := event88219
    frameStart := 88179 },
  { event := event88220
    frameStart := 88179 },
  { event := event88221
    frameStart := 88179 },
  { event := event88222
    frameStart := 88179 },
  { event := event88223
    frameStart := 88179 }
]

def eventLeaf5514 : Array AnnotatedEvent := #[
  { event := event88224
    frameStart := 88179 },
  { event := event88225
    frameStart := 88179 },
  { event := event88226
    frameStart := 88179 },
  { event := event88227
    frameStart := 88227 },
  { event := event88228
    frameStart := 88227 },
  { event := event88229
    frameStart := 88227 },
  { event := event88230
    frameStart := 88227 },
  { event := event88231
    frameStart := 88227 },
  { event := event88232
    frameStart := 88227 },
  { event := event88233
    frameStart := 88227 },
  { event := event88234
    frameStart := 88227 },
  { event := event88235
    frameStart := 88227 },
  { event := event88236
    frameStart := 88227 },
  { event := event88237
    frameStart := 88227 },
  { event := event88238
    frameStart := 88227 },
  { event := event88239
    frameStart := 88227 }
]

def eventLeaf5515 : Array AnnotatedEvent := #[
  { event := event88240
    frameStart := 88227 },
  { event := event88241
    frameStart := 88227 },
  { event := event88242
    frameStart := 88227 },
  { event := event88243
    frameStart := 88227 },
  { event := event88244
    frameStart := 88227 },
  { event := event88245
    frameStart := 88227 },
  { event := event88246
    frameStart := 88227 },
  { event := event88247
    frameStart := 88227 },
  { event := event88248
    frameStart := 88227 },
  { event := event88249
    frameStart := 88227 },
  { event := event88250
    frameStart := 88227 },
  { event := event88251
    frameStart := 88227 },
  { event := event88252
    frameStart := 88227 },
  { event := event88253
    frameStart := 88227 },
  { event := event88254
    frameStart := 88227 },
  { event := event88255
    frameStart := 88227 }
]

def eventLeaf5516 : Array AnnotatedEvent := #[
  { event := event88256
    frameStart := 88227 },
  { event := event88257
    frameStart := 88227 },
  { event := event88258
    frameStart := 88227 },
  { event := event88259
    frameStart := 88227 },
  { event := event88260
    frameStart := 88227 },
  { event := event88261
    frameStart := 88227 },
  { event := event88262
    frameStart := 88227 },
  { event := event88263
    frameStart := 88227 },
  { event := event88264
    frameStart := 88227 },
  { event := event88265
    frameStart := 88227 },
  { event := event88266
    frameStart := 88227 },
  { event := event88267
    frameStart := 88227 },
  { event := event88268
    frameStart := 88227 },
  { event := event88269
    frameStart := 88227 },
  { event := event88270
    frameStart := 88227 },
  { event := event88271
    frameStart := 88227 }
]

def eventLeaf5517 : Array AnnotatedEvent := #[
  { event := event88272
    frameStart := 88227 },
  { event := event88273
    frameStart := 88227 },
  { event := event88274
    frameStart := 88227 },
  { event := event88275
    frameStart := 88227 },
  { event := event88276
    frameStart := 88227 },
  { event := event88277
    frameStart := 88227 },
  { event := event88278
    frameStart := 88227 },
  { event := event88279
    frameStart := 88227 },
  { event := event88280
    frameStart := 88227 },
  { event := event88281
    frameStart := 88227 },
  { event := event88282
    frameStart := 88227 },
  { event := event88283
    frameStart := 88227 },
  { event := event88284
    frameStart := 88227 },
  { event := event88285
    frameStart := 88227 },
  { event := event88286
    frameStart := 88227 },
  { event := event88287
    frameStart := 88227 }
]

def eventLeaf5518 : Array AnnotatedEvent := #[
  { event := event88288
    frameStart := 88227 },
  { event := event88289
    frameStart := 88227 },
  { event := event88290
    frameStart := 88227 },
  { event := event88291
    frameStart := 88227 },
  { event := event88292
    frameStart := 88227 },
  { event := event88293
    frameStart := 88227 },
  { event := event88294
    frameStart := 88227 },
  { event := event88295
    frameStart := 88227 },
  { event := event88296
    frameStart := 88227 },
  { event := event88297
    frameStart := 88227 },
  { event := event88298
    frameStart := 88227 },
  { event := event88299
    frameStart := 88227 },
  { event := event88300
    frameStart := 88227 },
  { event := event88301
    frameStart := 88227 },
  { event := event88302
    frameStart := 88227 },
  { event := event88303
    frameStart := 88227 }
]

def eventLeaf5519 : Array AnnotatedEvent := #[
  { event := event88304
    frameStart := 88227 },
  { event := event88305
    frameStart := 88227 },
  { event := event88306
    frameStart := 88227 },
  { event := event88307
    frameStart := 88227 },
  { event := event88308
    frameStart := 88227 },
  { event := event88309
    frameStart := 88227 },
  { event := event88310
    frameStart := 88227 },
  { event := event88311
    frameStart := 88227 },
  { event := event88312
    frameStart := 88227 },
  { event := event88313
    frameStart := 88227 },
  { event := event88314
    frameStart := 88227 },
  { event := event88315
    frameStart := 88227 },
  { event := event88316
    frameStart := 88227 },
  { event := event88317
    frameStart := 88227 },
  { event := event88318
    frameStart := 88227 },
  { event := event88319
    frameStart := 88227 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events344
