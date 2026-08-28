import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events891

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event228096 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event228097 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 0 ⟨392⟩ 228096

def event228098 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 1 ⟨4990⟩ 228088

def event228099 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.sum [.predecessor 0 228097 .coefficient, .predecessor 1 228098 .coefficient])

def event228100 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.finite 655345)

def event228101 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 0 ⟨4992⟩ 228100

def event228102 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 1 ⟨5426⟩ 228086

def event228103 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.identity (.predecessor 1 228102 .coefficient))

def event228104 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.finite 655360)

def event228105 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24758⟩⟩) 0 ⟨5577⟩ 228104

def event228106 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24758⟩⟩) (.authority (.programFamilyFact))

def exact228107RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24758⟩⟩], []⟩, (1)⟩]

theorem exact228107RawTermsValid :
    exact228107RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228107 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24758⟩⟩) exact228107RawTerms (.finite 12) 228106 .exactZero (none)

def event228108 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53498⟩⟩) 0 ⟨5577⟩ 228104

def event228109 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53498⟩⟩) (.authority (.programFamilyFact))

def exact228110RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53498⟩⟩], []⟩, (1)⟩]

theorem exact228110RawTermsValid :
    exact228110RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228110 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53498⟩⟩) exact228110RawTerms (.finite 12) 228109 .exactZero (none)

def event228111 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53499⟩⟩) 0 ⟨53498⟩ 228110

def event228112 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53499⟩⟩) 1 ⟨24758⟩ 228107

def event228113 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53499⟩⟩) (.product (.predecessor 0 228111 .coefficient) (.predecessor 1 228112 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event228114 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53499⟩⟩, .operator (⟨228110, 0⟩, ⟨228107, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], []⟩, (1)⟩)

def exact228115RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], []⟩, (1)⟩]

theorem exact228115RawTermsValid :
    exact228115RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228115 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53499⟩⟩) exact228115RawTerms (.finite 144) 228113 .exactZero (none)

def event228116 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53500⟩⟩) 0 ⟨53499⟩ 228115

def event228117 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53500⟩⟩) (.identity (.predecessor 0 228116 .coefficient))

def event228118 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53500⟩⟩) (.finite 144)

def event228119 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54982⟩⟩) 0 ⟨53500⟩ 228118

def event228120 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54982⟩⟩) (.authority (.programFamilyFact))

def event228121 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨54982⟩⟩) (.finite 3720)

def event228122 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event228123 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54983⟩⟩) 0 ⟨7177⟩ 228122

def event228124 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54983⟩⟩) 1 ⟨54982⟩ 228121

def event228125 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54983⟩⟩) (.authority (.operator))

def exact228126RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54983⟩⟩]⟩, (1)⟩]

theorem exact228126RawTermsValid :
    exact228126RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228126 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54983⟩⟩) exact228126RawTerms .large 228125 .exactZero (none)

def event228127 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55488⟩⟩) 0 ⟨54983⟩ 228126

def event228128 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55488⟩⟩) (.authority (.operator))

def exact228129RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨55488⟩⟩]⟩, (1)⟩]

theorem exact228129RawTermsValid :
    exact228129RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228129 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55488⟩⟩) exact228129RawTerms (.finite 8192) 228128 .exactZero (none)

def event228130 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event228131 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event228132 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55262⟩⟩) 0 ⟨53500⟩ 228118

def event228133 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55262⟩⟩) 1 ⟨136⟩ 228131

def event228134 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55262⟩⟩) (.sum [.predecessor 0 228132 .coefficient, .predecessor 1 228133 .coefficient])

def event228135 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨55262⟩⟩) (.finite 144)

def event228136 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55263⟩⟩) 0 ⟨55262⟩ 228135

def event228137 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55263⟩⟩) (.identity (.predecessor 0 228136 .coefficient))

def exact228138RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], []⟩, (1)⟩]

theorem exact228138RawTermsValid :
    exact228138RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228138 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55263⟩⟩) exact228138RawTerms (.finite 144) 228137 .exactZero (none)

def event228139 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact228140RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact228140RawTermsValid :
    exact228140RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228140 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact228140RawTerms .large 228139 .exactZero (none)

def event228141 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55264⟩⟩) 0 ⟨6908⟩ 228140

def event228142 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55264⟩⟩) 1 ⟨55263⟩ 228138

def event228143 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55264⟩⟩) (.product (.predecessor 0 228141 .coefficient) (.predecessor 1 228142 .coefficient) (⟨false, false, none, none, none⟩))

def event228144 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55264⟩⟩, .operator (⟨228140, 0⟩, ⟨228138, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact228145RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact228145RawTermsValid :
    exact228145RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228145 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55264⟩⟩) exact228145RawTerms .large 228143 .exactZero (none)

def event228146 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event228147 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event228148 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 228122

def event228149 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact228150RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact228150RawTermsValid :
    exact228150RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228150 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact228150RawTerms .large 228149 .exactZero (none)

def event228151 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7272⟩⟩) 0 ⟨7178⟩ 228150

def event228152 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7272⟩⟩) (.identity (.predecessor 0 228151 .coefficient))

def exact228153RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7272⟩⟩]⟩, (1)⟩]

theorem exact228153RawTermsValid :
    exact228153RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228153 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7272⟩⟩) exact228153RawTerms .large 228152 .exactZero (none)

def event228154 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9529⟩⟩) 0 ⟨7272⟩ 228153

def event228155 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9529⟩⟩) (.authority (.operator))

def exact228156RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩, (1)⟩]

theorem exact228156RawTermsValid :
    exact228156RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228156 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9529⟩⟩) exact228156RawTerms (.finite 8192) 228155 .exactZero (none)

def event228157 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9530⟩⟩) 0 ⟨9529⟩ 228156

def event228158 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9530⟩⟩) 1 ⟨2370⟩ 228147

def event228159 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9530⟩⟩) (.scale (.predecessor 0 228157 .coefficient) (.value (.predecessor 1 228158 .coefficient)))

def exact228160RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩, (1)⟩]

theorem exact228160RawTermsValid :
    exact228160RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228160 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9530⟩⟩) exact228160RawTerms (.finite 8192) 228159 .exactZero (none)

def event228161 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7289⟩⟩) 0 ⟨7178⟩ 228150

def event228162 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7289⟩⟩) (.identity (.predecessor 0 228161 .coefficient))

def exact228163RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩]⟩, (1)⟩]

theorem exact228163RawTermsValid :
    exact228163RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228163 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7289⟩⟩) exact228163RawTerms .large 228162 .exactZero (none)

def event228164 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9531⟩⟩) 0 ⟨7289⟩ 228163

def event228165 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9531⟩⟩) 1 ⟨9530⟩ 228160

def event228166 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9531⟩⟩) (.product (.predecessor 0 228164 .coefficient) (.predecessor 1 228165 .coefficient) (⟨false, false, none, none, none⟩))

def event228167 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9531⟩⟩, .operator (⟨228163, 0⟩, ⟨228160, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩, (1)⟩)

def exact228168RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩, (1)⟩]

theorem exact228168RawTermsValid :
    exact228168RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228168 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9531⟩⟩) exact228168RawTerms .large 228166 .exactZero (none)

def event228169 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55265⟩⟩) 0 ⟨9531⟩ 228168

def event228170 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55265⟩⟩) 1 ⟨55264⟩ 228145

def event228171 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55265⟩⟩) (.sum [.predecessor 0 228169 .coefficient, .predecessor 1 228170 .coefficient])

def exact228172RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228172RawTermsValid :
    exact228172RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228172 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55265⟩⟩) exact228172RawTerms .large 228171 .exactZero (none)

def event228173 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55491⟩⟩) 0 ⟨55265⟩ 228172

def event228174 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55491⟩⟩) 1 ⟨55488⟩ 228129

def event228175 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55491⟩⟩) (.product (.predecessor 0 228173 .coefficient) (.predecessor 1 228174 .coefficient) (⟨false, false, none, none, none⟩))

def event228176 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55491⟩⟩, .operator (⟨228172, 0⟩, ⟨228129, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55488⟩⟩]⟩, (1)⟩)

def event228177 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55491⟩⟩, .operator (⟨228172, 1⟩, ⟨228129, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55488⟩⟩]⟩, (-1)⟩)

def event228178 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨55491⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55488⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55488⟩⟩) ⟨54983⟩ 228126)

def event228179 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55491⟩⟩, .relation 228178 0, ⟨[⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨54983⟩⟩]⟩, (-1)⟩)

def exact228180RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55488⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨54983⟩⟩]⟩, (-1)⟩]

theorem exact228180RawTermsValid :
    exact228180RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228180 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55491⟩⟩) exact228180RawTerms .large 228175 .exactZero (none)

def event228181 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53860⟩⟩) 0 ⟨53500⟩ 228118

def event228182 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53860⟩⟩) (.authority (.programFamilyFact))

def exact228183RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53860⟩⟩], []⟩, (1)⟩]

theorem exact228183RawTermsValid :
    exact228183RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228183 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53860⟩⟩) exact228183RawTerms (.finite 12) 228182 .exactZero (none)

def event228184 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53862⟩⟩) 0 ⟨6908⟩ 228140

def event228185 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53862⟩⟩) 1 ⟨53860⟩ 228183

def event228186 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53862⟩⟩) (.product (.predecessor 0 228184 .coefficient) (.predecessor 1 228185 .coefficient) (⟨false, true, none, none, some 1⟩))

def event228187 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53862⟩⟩, .operator (⟨228140, 0⟩, ⟨228183, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact228188RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact228188RawTermsValid :
    exact228188RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228188 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53862⟩⟩) exact228188RawTerms .large 228186 .exactZero (none)

def event228189 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7184⟩⟩) 0 ⟨7177⟩ 228122

def event228190 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7184⟩⟩) (.authority (.operator))

def exact228191RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩]

theorem exact228191RawTermsValid :
    exact228191RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228191 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7184⟩⟩) exact228191RawTerms .large 228190 .exactZero (none)

def event228192 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53863⟩⟩) 0 ⟨7184⟩ 228191

def event228193 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53863⟩⟩) 1 ⟨53862⟩ 228188

def event228194 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53863⟩⟩) (.sum [.predecessor 0 228192 .coefficient, .predecessor 1 228193 .coefficient])

def exact228195RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228195RawTermsValid :
    exact228195RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228195 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53863⟩⟩) exact228195RawTerms .large 228194 .exactZero (none)

def event228196 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55492⟩⟩) 0 ⟨53863⟩ 228195

def event228197 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55492⟩⟩) 1 ⟨55491⟩ 228180

def event228198 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55492⟩⟩) (.sum [.predecessor 0 228196 .coefficient, .predecessor 1 228197 .coefficient])

def exact228199RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55488⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨54983⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228199RawTermsValid :
    exact228199RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228199 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55492⟩⟩) exact228199RawTerms .large 228198 .exactZero (none)

def event228200 : Event := .preFoldPolynomial 228199 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55488⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨54983⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact228201RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55488⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨54983⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event228201 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨55492⟩⟩) 228200 exact228201RawTerms .large 228198 .exactZero (none)

def event228202 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨53500⟩⟩) ⟨⟨63⟩, ⟨41⟩, ⟨135⟩⟩ ⟨228036, 228202⟩

def event228203 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨54422⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54419⟩⟩]⟩) (1) 0 2 (.universal 228202 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54419⟩⟩]⟩) (none) 228201)

def event228204 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54422⟩⟩, .relation 228203 0, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩)

def event228205 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54422⟩⟩, .relation 228203 1, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55488⟩⟩]⟩, (-1)⟩)

def event228206 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54422⟩⟩, .relation 228203 2, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨54983⟩⟩]⟩, (1)⟩)

def event228207 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54422⟩⟩, .relation 228203 3, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact228208RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55488⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨54983⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228208RawTermsValid :
    exact228208RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228208 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54422⟩⟩) exact228208RawTerms .large 228032 (.finite 202072841853861888) (some (228034))

def event228209 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55490⟩⟩) 0 ⟨54422⟩ 228208

def event228210 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55490⟩⟩) 1 ⟨55489⟩ 228022

def event228211 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55490⟩⟩) (.sum [.predecessor 0 228209 .coefficient, .predecessor 1 228210 .coefficient])

def event228212 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55490⟩⟩, .operator (⟨228208, 2⟩, ⟨228022, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], [⟨.program ⟨257⟩, ⟨54983⟩⟩]⟩, (-1)⟩)

def event228213 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55490⟩⟩, .operator (⟨228208, 1⟩, ⟨228022, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55488⟩⟩]⟩, (1)⟩)

def event228214 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55490⟩⟩) (.sum [.result 228208 .summary, .result 228022 .summary])

def exact228215RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact228215RawTermsValid :
    exact228215RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228215 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55490⟩⟩) exact228215RawTerms .large 228211 (.finite 2997907760060573155328) (some (228214))

def event228216 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55903⟩⟩) 0 ⟨55490⟩ 228215

def event228217 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55903⟩⟩) 1 ⟨55901⟩ 227938

def event228218 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55903⟩⟩) (.product (.predecessor 0 228216 .coefficient) (.predecessor 1 228217 .coefficient) (⟨false, false, none, none, none⟩))

def event228219 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55903⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨55901⟩⟩]⟩) [⟨.result 227938 .coefficient, false, none⟩])

def event228220 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55903⟩⟩) (.product (.result 228215 .summary) (.transfer 228219) (⟨false, false, none, none, none⟩))

def event228221 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55903⟩⟩, .operator (⟨228215, 0⟩, ⟨227938, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55901⟩⟩]⟩, (1)⟩)

def event228222 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55903⟩⟩, .operator (⟨228215, 1⟩, ⟨227938, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55901⟩⟩]⟩, (-1)⟩)

def event228223 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨55903⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55901⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55901⟩⟩) ⟨55132⟩ 227935)

def event228224 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨55903⟩⟩, .relation 228223 0, ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨55132⟩⟩]⟩, (-1)⟩)

def exact228225RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55901⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨53860⟩⟩], [⟨.program ⟨257⟩, ⟨55132⟩⟩]⟩, (-1)⟩]

theorem exact228225RawTermsValid :
    exact228225RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228225 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55903⟩⟩) exact228225RawTerms .large 228218 (.finite 32189789464711941702873220382720) (some (228220))

def event228226 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54716⟩⟩) 0 ⟨53861⟩ 10859

def event228227 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54716⟩⟩) (.authority (.relationPreimageSource ⟨68⟩))

def exact228228RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54716⟩⟩]⟩, (1)⟩]

theorem exact228228RawTermsValid :
    exact228228RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228228 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54716⟩⟩) exact228228RawTerms (.finite 5647228698) 228227 .exactZero (none)

def event228229 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54718⟩⟩) 0 ⟨54716⟩ 228228

def event228230 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54718⟩⟩) 1 ⟨2370⟩ 4

def event228231 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54718⟩⟩) (.scale (.predecessor 0 228229 .coefficient) (.value (.predecessor 1 228230 .coefficient)))

def exact228232RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54716⟩⟩]⟩, (1)⟩]

theorem exact228232RawTermsValid :
    exact228232RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228232 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54718⟩⟩) exact228232RawTerms (.finite 5647228698) 228231 .exactZero (none)

def event228233 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54719⟩⟩) 0 ⟨5581⟩ 222245

def event228234 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54719⟩⟩) 1 ⟨54718⟩ 228232

def event228235 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54719⟩⟩) (.product (.predecessor 0 228233 .coefficient) (.predecessor 1 228234 .coefficient) (⟨false, false, none, none, none⟩))

def event228236 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54719⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨54716⟩⟩]⟩) [⟨.result 228228 .coefficient, false, none⟩])

def event228237 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54719⟩⟩) (.product (.result 222245 .summary) (.transfer 228236) (⟨false, false, none, none, none⟩))

def event228238 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54719⟩⟩, .operator (⟨222245, 0⟩, ⟨228232, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54716⟩⟩]⟩, (1)⟩)

def event228239 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨54717⟩⟩)

def event228240 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event228241 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event228242 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.authority (.operator))

def event228243 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.finite 5)

def event228244 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event228245 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event228246 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event228247 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event228248 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 228247

def event228249 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 228245

def event228250 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 228248 .coefficient) (.value (.predecessor 1 228249 .coefficient)))

def event228251 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event228252 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 0 ⟨392⟩ 228251

def event228253 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 1 ⟨4990⟩ 228243

def event228254 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.sum [.predecessor 0 228252 .coefficient, .predecessor 1 228253 .coefficient])

def event228255 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.finite 655345)

def event228256 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 0 ⟨4992⟩ 228255

def event228257 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 1 ⟨5426⟩ 228241

def event228258 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.identity (.predecessor 1 228257 .coefficient))

def event228259 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.finite 655360)

def event228260 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24758⟩⟩) 0 ⟨5577⟩ 228259

def event228261 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24758⟩⟩) (.authority (.programFamilyFact))

def exact228262RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24758⟩⟩], []⟩, (1)⟩]

theorem exact228262RawTermsValid :
    exact228262RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228262 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24758⟩⟩) exact228262RawTerms (.finite 12) 228261 .exactZero (none)

def event228263 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53498⟩⟩) 0 ⟨5577⟩ 228259

def event228264 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53498⟩⟩) (.authority (.programFamilyFact))

def exact228265RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53498⟩⟩], []⟩, (1)⟩]

theorem exact228265RawTermsValid :
    exact228265RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228265 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53498⟩⟩) exact228265RawTerms (.finite 12) 228264 .exactZero (none)

def event228266 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53499⟩⟩) 0 ⟨53498⟩ 228265

def event228267 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53499⟩⟩) 1 ⟨24758⟩ 228262

def event228268 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53499⟩⟩) (.product (.predecessor 0 228266 .coefficient) (.predecessor 1 228267 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event228269 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53499⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], []⟩) [⟨.result 228265 .coefficient, true, some 1⟩, ⟨.result 228262 .coefficient, true, some 1⟩])

def event228270 : Event := .survivorFold (1) 228269

def exact228271RawTerms : List Term := []

theorem exact228271RawTermsValid :
    exact228271RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228271 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53499⟩⟩) exact228271RawTerms (.finite 144) 228268 (.finite 144) (some (228269))

def event228272 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53500⟩⟩) 0 ⟨53499⟩ 228271

def event228273 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53500⟩⟩) (.identity (.predecessor 0 228272 .coefficient))

def event228274 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53500⟩⟩) (.finite 144)

def event228275 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53860⟩⟩) 0 ⟨53500⟩ 228274

def event228276 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53860⟩⟩) (.authority (.programFamilyFact))

def exact228277RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53860⟩⟩], []⟩, (1)⟩]

theorem exact228277RawTermsValid :
    exact228277RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228277 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53860⟩⟩) exact228277RawTerms (.finite 12) 228276 .exactZero (none)

def event228278 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53861⟩⟩) 0 ⟨53860⟩ 228277

def event228279 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53861⟩⟩) (.identity (.predecessor 0 228278 .coefficient))

def event228280 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53861⟩⟩) (.finite 12)

def event228281 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54716⟩⟩) 0 ⟨53861⟩ 228280

def event228282 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54716⟩⟩) (.authority (.relationPreimageSource ⟨68⟩))

def exact228283RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨54716⟩⟩]⟩, (1)⟩]

theorem exact228283RawTermsValid :
    exact228283RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228283 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54716⟩⟩) exact228283RawTerms (.finite 5647228698) 228282 .exactZero (none)

def event228284 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact228285RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact228285RawTermsValid :
    exact228285RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228285 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact228285RawTerms .large 228284 .exactZero (none)

def event228286 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54717⟩⟩) 0 ⟨35⟩ 228285

def event228287 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54717⟩⟩) 1 ⟨54716⟩ 228283

def event228288 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54717⟩⟩) (.product (.predecessor 0 228286 .coefficient) (.predecessor 1 228287 .coefficient) (⟨false, false, none, none, none⟩))

def event228289 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54717⟩⟩, .operator (⟨228285, 0⟩, ⟨228283, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54716⟩⟩]⟩, (1)⟩)

def exact228290RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54716⟩⟩]⟩, (1)⟩]

theorem exact228290RawTermsValid :
    exact228290RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228290 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54717⟩⟩) exact228290RawTerms .large 228288 .exactZero (none)

def event228291 : Event := .preFoldPolynomial 228290 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54716⟩⟩]⟩, (1)⟩] .exactZero none

def exact228292RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54716⟩⟩]⟩, (1)⟩]

def event228292 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨54717⟩⟩) 228291 exact228292RawTerms .large 228288 .exactZero (none)

def event228293 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨55906⟩⟩)

def event228294 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event228295 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event228296 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.authority (.operator))

def event228297 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4990⟩⟩) (.finite 5)

def event228298 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event228299 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event228300 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event228301 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event228302 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 228301

def event228303 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 228299

def event228304 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 228302 .coefficient) (.value (.predecessor 1 228303 .coefficient)))

def event228305 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event228306 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 0 ⟨392⟩ 228305

def event228307 : Event := .predecessor (⟨.program ⟨257⟩, ⟨4992⟩⟩) 1 ⟨4990⟩ 228297

def event228308 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.sum [.predecessor 0 228306 .coefficient, .predecessor 1 228307 .coefficient])

def event228309 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨4992⟩⟩) (.finite 655345)

def event228310 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 0 ⟨4992⟩ 228309

def event228311 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5577⟩⟩) 1 ⟨5426⟩ 228295

def event228312 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.identity (.predecessor 1 228311 .coefficient))

def event228313 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5577⟩⟩) (.finite 655360)

def event228314 : Event := .predecessor (⟨.program ⟨257⟩, ⟨24758⟩⟩) 0 ⟨5577⟩ 228313

def event228315 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨24758⟩⟩) (.authority (.programFamilyFact))

def exact228316RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24758⟩⟩], []⟩, (1)⟩]

theorem exact228316RawTermsValid :
    exact228316RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228316 : Event := .resultExact (⟨.program ⟨257⟩, ⟨24758⟩⟩) exact228316RawTerms (.finite 12) 228315 .exactZero (none)

def event228317 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53498⟩⟩) 0 ⟨5577⟩ 228313

def event228318 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53498⟩⟩) (.authority (.programFamilyFact))

def exact228319RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53498⟩⟩], []⟩, (1)⟩]

theorem exact228319RawTermsValid :
    exact228319RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228319 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53498⟩⟩) exact228319RawTerms (.finite 12) 228318 .exactZero (none)

def event228320 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53499⟩⟩) 0 ⟨53498⟩ 228319

def event228321 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53499⟩⟩) 1 ⟨24758⟩ 228316

def event228322 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53499⟩⟩) (.product (.predecessor 0 228320 .coefficient) (.predecessor 1 228321 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event228323 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨53499⟩⟩, .operator (⟨228319, 0⟩, ⟨228316, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], []⟩, (1)⟩)

def exact228324RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨24758⟩⟩, ⟨.program ⟨257⟩, ⟨53498⟩⟩], []⟩, (1)⟩]

theorem exact228324RawTermsValid :
    exact228324RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228324 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53499⟩⟩) exact228324RawTerms (.finite 144) 228322 .exactZero (none)

def event228325 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53500⟩⟩) 0 ⟨53499⟩ 228324

def event228326 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53500⟩⟩) (.identity (.predecessor 0 228325 .coefficient))

def event228327 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53500⟩⟩) (.finite 144)

def event228328 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53860⟩⟩) 0 ⟨53500⟩ 228327

def event228329 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53860⟩⟩) (.authority (.programFamilyFact))

def exact228330RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨53860⟩⟩], []⟩, (1)⟩]

theorem exact228330RawTermsValid :
    exact228330RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228330 : Event := .resultExact (⟨.program ⟨257⟩, ⟨53860⟩⟩) exact228330RawTerms (.finite 12) 228329 .exactZero (none)

def event228331 : Event := .predecessor (⟨.program ⟨257⟩, ⟨53861⟩⟩) 0 ⟨53860⟩ 228330

def event228332 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨53861⟩⟩) (.identity (.predecessor 0 228331 .coefficient))

def event228333 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨53861⟩⟩) (.finite 12)

def event228334 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55130⟩⟩) 0 ⟨53861⟩ 228333

def event228335 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55130⟩⟩) (.authority (.programFamilyFact))

def event228336 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨55130⟩⟩) (.finite 3720)

def event228337 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event228338 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55132⟩⟩) 0 ⟨7177⟩ 228337

def event228339 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55132⟩⟩) 1 ⟨55130⟩ 228336

def event228340 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55132⟩⟩) (.authority (.operator))

def exact228341RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨55132⟩⟩]⟩, (1)⟩]

theorem exact228341RawTermsValid :
    exact228341RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228341 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55132⟩⟩) exact228341RawTerms .large 228340 .exactZero (none)

def event228342 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55901⟩⟩) 0 ⟨55132⟩ 228341

def event228343 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55901⟩⟩) (.authority (.operator))

def exact228344RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨55901⟩⟩]⟩, (1)⟩]

theorem exact228344RawTermsValid :
    exact228344RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event228344 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55901⟩⟩) exact228344RawTerms (.finite 8192) 228343 .exactZero (none)

def event228345 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event228346 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event228347 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55342⟩⟩) 0 ⟨53861⟩ 228333

def event228348 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55342⟩⟩) 1 ⟨136⟩ 228346

def event228349 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55342⟩⟩) (.sum [.predecessor 0 228347 .coefficient, .predecessor 1 228348 .coefficient])

def event228350 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨55342⟩⟩) (.finite 12)

def event228351 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55343⟩⟩) 0 ⟨55342⟩ 228350

def eventLeaf14256 : Array AnnotatedEvent := #[
  { event := event228096
    frameStart := 228084 },
  { event := event228097
    frameStart := 228084 },
  { event := event228098
    frameStart := 228084 },
  { event := event228099
    frameStart := 228084 },
  { event := event228100
    frameStart := 228084 },
  { event := event228101
    frameStart := 228084 },
  { event := event228102
    frameStart := 228084 },
  { event := event228103
    frameStart := 228084 },
  { event := event228104
    frameStart := 228084 },
  { event := event228105
    frameStart := 228084 },
  { event := event228106
    frameStart := 228084 },
  { event := event228107
    frameStart := 228084 },
  { event := event228108
    frameStart := 228084 },
  { event := event228109
    frameStart := 228084 },
  { event := event228110
    frameStart := 228084 },
  { event := event228111
    frameStart := 228084 }
]

def eventLeaf14257 : Array AnnotatedEvent := #[
  { event := event228112
    frameStart := 228084 },
  { event := event228113
    frameStart := 228084 },
  { event := event228114
    frameStart := 228084 },
  { event := event228115
    frameStart := 228084 },
  { event := event228116
    frameStart := 228084 },
  { event := event228117
    frameStart := 228084 },
  { event := event228118
    frameStart := 228084 },
  { event := event228119
    frameStart := 228084 },
  { event := event228120
    frameStart := 228084 },
  { event := event228121
    frameStart := 228084 },
  { event := event228122
    frameStart := 228084 },
  { event := event228123
    frameStart := 228084 },
  { event := event228124
    frameStart := 228084 },
  { event := event228125
    frameStart := 228084 },
  { event := event228126
    frameStart := 228084 },
  { event := event228127
    frameStart := 228084 }
]

def eventLeaf14258 : Array AnnotatedEvent := #[
  { event := event228128
    frameStart := 228084 },
  { event := event228129
    frameStart := 228084 },
  { event := event228130
    frameStart := 228084 },
  { event := event228131
    frameStart := 228084 },
  { event := event228132
    frameStart := 228084 },
  { event := event228133
    frameStart := 228084 },
  { event := event228134
    frameStart := 228084 },
  { event := event228135
    frameStart := 228084 },
  { event := event228136
    frameStart := 228084 },
  { event := event228137
    frameStart := 228084 },
  { event := event228138
    frameStart := 228084 },
  { event := event228139
    frameStart := 228084 },
  { event := event228140
    frameStart := 228084 },
  { event := event228141
    frameStart := 228084 },
  { event := event228142
    frameStart := 228084 },
  { event := event228143
    frameStart := 228084 }
]

def eventLeaf14259 : Array AnnotatedEvent := #[
  { event := event228144
    frameStart := 228084 },
  { event := event228145
    frameStart := 228084 },
  { event := event228146
    frameStart := 228084 },
  { event := event228147
    frameStart := 228084 },
  { event := event228148
    frameStart := 228084 },
  { event := event228149
    frameStart := 228084 },
  { event := event228150
    frameStart := 228084 },
  { event := event228151
    frameStart := 228084 },
  { event := event228152
    frameStart := 228084 },
  { event := event228153
    frameStart := 228084 },
  { event := event228154
    frameStart := 228084 },
  { event := event228155
    frameStart := 228084 },
  { event := event228156
    frameStart := 228084 },
  { event := event228157
    frameStart := 228084 },
  { event := event228158
    frameStart := 228084 },
  { event := event228159
    frameStart := 228084 }
]

def eventLeaf14260 : Array AnnotatedEvent := #[
  { event := event228160
    frameStart := 228084 },
  { event := event228161
    frameStart := 228084 },
  { event := event228162
    frameStart := 228084 },
  { event := event228163
    frameStart := 228084 },
  { event := event228164
    frameStart := 228084 },
  { event := event228165
    frameStart := 228084 },
  { event := event228166
    frameStart := 228084 },
  { event := event228167
    frameStart := 228084 },
  { event := event228168
    frameStart := 228084 },
  { event := event228169
    frameStart := 228084 },
  { event := event228170
    frameStart := 228084 },
  { event := event228171
    frameStart := 228084 },
  { event := event228172
    frameStart := 228084 },
  { event := event228173
    frameStart := 228084 },
  { event := event228174
    frameStart := 228084 },
  { event := event228175
    frameStart := 228084 }
]

def eventLeaf14261 : Array AnnotatedEvent := #[
  { event := event228176
    frameStart := 228084 },
  { event := event228177
    frameStart := 228084 },
  { event := event228178
    frameStart := 228084 },
  { event := event228179
    frameStart := 228084 },
  { event := event228180
    frameStart := 228084 },
  { event := event228181
    frameStart := 228084 },
  { event := event228182
    frameStart := 228084 },
  { event := event228183
    frameStart := 228084 },
  { event := event228184
    frameStart := 228084 },
  { event := event228185
    frameStart := 228084 },
  { event := event228186
    frameStart := 228084 },
  { event := event228187
    frameStart := 228084 },
  { event := event228188
    frameStart := 228084 },
  { event := event228189
    frameStart := 228084 },
  { event := event228190
    frameStart := 228084 },
  { event := event228191
    frameStart := 228084 }
]

def eventLeaf14262 : Array AnnotatedEvent := #[
  { event := event228192
    frameStart := 228084 },
  { event := event228193
    frameStart := 228084 },
  { event := event228194
    frameStart := 228084 },
  { event := event228195
    frameStart := 228084 },
  { event := event228196
    frameStart := 228084 },
  { event := event228197
    frameStart := 228084 },
  { event := event228198
    frameStart := 228084 },
  { event := event228199
    frameStart := 228084 },
  { event := event228200
    frameStart := 228084 },
  { event := event228201
    frameStart := 228084 },
  { event := event228202
    frameStart := 0 },
  { event := event228203
    frameStart := 0 },
  { event := event228204
    frameStart := 0 },
  { event := event228205
    frameStart := 0 },
  { event := event228206
    frameStart := 0 },
  { event := event228207
    frameStart := 0 }
]

def eventLeaf14263 : Array AnnotatedEvent := #[
  { event := event228208
    frameStart := 0 },
  { event := event228209
    frameStart := 0 },
  { event := event228210
    frameStart := 0 },
  { event := event228211
    frameStart := 0 },
  { event := event228212
    frameStart := 0 },
  { event := event228213
    frameStart := 0 },
  { event := event228214
    frameStart := 0 },
  { event := event228215
    frameStart := 0 },
  { event := event228216
    frameStart := 0 },
  { event := event228217
    frameStart := 0 },
  { event := event228218
    frameStart := 0 },
  { event := event228219
    frameStart := 0 },
  { event := event228220
    frameStart := 0 },
  { event := event228221
    frameStart := 0 },
  { event := event228222
    frameStart := 0 },
  { event := event228223
    frameStart := 0 }
]

def eventLeaf14264 : Array AnnotatedEvent := #[
  { event := event228224
    frameStart := 0 },
  { event := event228225
    frameStart := 0 },
  { event := event228226
    frameStart := 0 },
  { event := event228227
    frameStart := 0 },
  { event := event228228
    frameStart := 0 },
  { event := event228229
    frameStart := 0 },
  { event := event228230
    frameStart := 0 },
  { event := event228231
    frameStart := 0 },
  { event := event228232
    frameStart := 0 },
  { event := event228233
    frameStart := 0 },
  { event := event228234
    frameStart := 0 },
  { event := event228235
    frameStart := 0 },
  { event := event228236
    frameStart := 0 },
  { event := event228237
    frameStart := 0 },
  { event := event228238
    frameStart := 0 },
  { event := event228239
    frameStart := 228239 }
]

def eventLeaf14265 : Array AnnotatedEvent := #[
  { event := event228240
    frameStart := 228239 },
  { event := event228241
    frameStart := 228239 },
  { event := event228242
    frameStart := 228239 },
  { event := event228243
    frameStart := 228239 },
  { event := event228244
    frameStart := 228239 },
  { event := event228245
    frameStart := 228239 },
  { event := event228246
    frameStart := 228239 },
  { event := event228247
    frameStart := 228239 },
  { event := event228248
    frameStart := 228239 },
  { event := event228249
    frameStart := 228239 },
  { event := event228250
    frameStart := 228239 },
  { event := event228251
    frameStart := 228239 },
  { event := event228252
    frameStart := 228239 },
  { event := event228253
    frameStart := 228239 },
  { event := event228254
    frameStart := 228239 },
  { event := event228255
    frameStart := 228239 }
]

def eventLeaf14266 : Array AnnotatedEvent := #[
  { event := event228256
    frameStart := 228239 },
  { event := event228257
    frameStart := 228239 },
  { event := event228258
    frameStart := 228239 },
  { event := event228259
    frameStart := 228239 },
  { event := event228260
    frameStart := 228239 },
  { event := event228261
    frameStart := 228239 },
  { event := event228262
    frameStart := 228239 },
  { event := event228263
    frameStart := 228239 },
  { event := event228264
    frameStart := 228239 },
  { event := event228265
    frameStart := 228239 },
  { event := event228266
    frameStart := 228239 },
  { event := event228267
    frameStart := 228239 },
  { event := event228268
    frameStart := 228239 },
  { event := event228269
    frameStart := 228239 },
  { event := event228270
    frameStart := 228239 },
  { event := event228271
    frameStart := 228239 }
]

def eventLeaf14267 : Array AnnotatedEvent := #[
  { event := event228272
    frameStart := 228239 },
  { event := event228273
    frameStart := 228239 },
  { event := event228274
    frameStart := 228239 },
  { event := event228275
    frameStart := 228239 },
  { event := event228276
    frameStart := 228239 },
  { event := event228277
    frameStart := 228239 },
  { event := event228278
    frameStart := 228239 },
  { event := event228279
    frameStart := 228239 },
  { event := event228280
    frameStart := 228239 },
  { event := event228281
    frameStart := 228239 },
  { event := event228282
    frameStart := 228239 },
  { event := event228283
    frameStart := 228239 },
  { event := event228284
    frameStart := 228239 },
  { event := event228285
    frameStart := 228239 },
  { event := event228286
    frameStart := 228239 },
  { event := event228287
    frameStart := 228239 }
]

def eventLeaf14268 : Array AnnotatedEvent := #[
  { event := event228288
    frameStart := 228239 },
  { event := event228289
    frameStart := 228239 },
  { event := event228290
    frameStart := 228239 },
  { event := event228291
    frameStart := 228239 },
  { event := event228292
    frameStart := 228239 },
  { event := event228293
    frameStart := 228293 },
  { event := event228294
    frameStart := 228293 },
  { event := event228295
    frameStart := 228293 },
  { event := event228296
    frameStart := 228293 },
  { event := event228297
    frameStart := 228293 },
  { event := event228298
    frameStart := 228293 },
  { event := event228299
    frameStart := 228293 },
  { event := event228300
    frameStart := 228293 },
  { event := event228301
    frameStart := 228293 },
  { event := event228302
    frameStart := 228293 },
  { event := event228303
    frameStart := 228293 }
]

def eventLeaf14269 : Array AnnotatedEvent := #[
  { event := event228304
    frameStart := 228293 },
  { event := event228305
    frameStart := 228293 },
  { event := event228306
    frameStart := 228293 },
  { event := event228307
    frameStart := 228293 },
  { event := event228308
    frameStart := 228293 },
  { event := event228309
    frameStart := 228293 },
  { event := event228310
    frameStart := 228293 },
  { event := event228311
    frameStart := 228293 },
  { event := event228312
    frameStart := 228293 },
  { event := event228313
    frameStart := 228293 },
  { event := event228314
    frameStart := 228293 },
  { event := event228315
    frameStart := 228293 },
  { event := event228316
    frameStart := 228293 },
  { event := event228317
    frameStart := 228293 },
  { event := event228318
    frameStart := 228293 },
  { event := event228319
    frameStart := 228293 }
]

def eventLeaf14270 : Array AnnotatedEvent := #[
  { event := event228320
    frameStart := 228293 },
  { event := event228321
    frameStart := 228293 },
  { event := event228322
    frameStart := 228293 },
  { event := event228323
    frameStart := 228293 },
  { event := event228324
    frameStart := 228293 },
  { event := event228325
    frameStart := 228293 },
  { event := event228326
    frameStart := 228293 },
  { event := event228327
    frameStart := 228293 },
  { event := event228328
    frameStart := 228293 },
  { event := event228329
    frameStart := 228293 },
  { event := event228330
    frameStart := 228293 },
  { event := event228331
    frameStart := 228293 },
  { event := event228332
    frameStart := 228293 },
  { event := event228333
    frameStart := 228293 },
  { event := event228334
    frameStart := 228293 },
  { event := event228335
    frameStart := 228293 }
]

def eventLeaf14271 : Array AnnotatedEvent := #[
  { event := event228336
    frameStart := 228293 },
  { event := event228337
    frameStart := 228293 },
  { event := event228338
    frameStart := 228293 },
  { event := event228339
    frameStart := 228293 },
  { event := event228340
    frameStart := 228293 },
  { event := event228341
    frameStart := 228293 },
  { event := event228342
    frameStart := 228293 },
  { event := event228343
    frameStart := 228293 },
  { event := event228344
    frameStart := 228293 },
  { event := event228345
    frameStart := 228293 },
  { event := event228346
    frameStart := 228293 },
  { event := event228347
    frameStart := 228293 },
  { event := event228348
    frameStart := 228293 },
  { event := event228349
    frameStart := 228293 },
  { event := event228350
    frameStart := 228293 },
  { event := event228351
    frameStart := 228293 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events891
