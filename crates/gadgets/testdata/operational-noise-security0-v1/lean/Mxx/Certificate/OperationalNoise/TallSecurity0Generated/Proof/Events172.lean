import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events172

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event44032 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25000⟩⟩, .operator (⟨44028, 2⟩, ⟨43842, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], [⟨.program ⟨214⟩, ⟨23000⟩⟩]⟩, (-1)⟩)

def event44033 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25000⟩⟩, .operator (⟨44028, 1⟩, ⟨43842, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24998⟩⟩]⟩, (1)⟩)

def event44034 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25000⟩⟩) (.sum [.result 44028 .summary, .result 43842 .summary])

def exact44035RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44035RawTermsValid :
    exact44035RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44035 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25000⟩⟩) exact44035RawTerms .large 44031 (.finite 352014917316608) (some (44034))

def event44036 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26592⟩⟩) 0 ⟨25000⟩ 44035

def event44037 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26592⟩⟩) 1 ⟨26590⟩ 43758

def event44038 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26592⟩⟩) (.product (.predecessor 0 44036 .coefficient) (.predecessor 1 44037 .coefficient) (⟨false, false, none, none, none⟩))

def event44039 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26592⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨26590⟩⟩]⟩) [⟨.result 43758 .coefficient, false, none⟩])

def event44040 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26592⟩⟩) (.product (.result 44035 .summary) (.transfer 44039) (⟨false, false, none, none, none⟩))

def event44041 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26592⟩⟩, .operator (⟨44035, 0⟩, ⟨43758, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26590⟩⟩]⟩, (1)⟩)

def event44042 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26592⟩⟩, .operator (⟨44035, 1⟩, ⟨43758, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26590⟩⟩]⟩, (-1)⟩)

def event44043 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26592⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26590⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26590⟩⟩) ⟨23790⟩ 43755)

def event44044 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26592⟩⟩, .relation 44043 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨23790⟩⟩]⟩, (-1)⟩)

def exact44045RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26590⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨23790⟩⟩]⟩, (-1)⟩]

theorem exact44045RawTermsValid :
    exact44045RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44045 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26592⟩⟩) exact44045RawTerms .large 44038 (.finite 1291900378790628425728) (some (44040))

def event44046 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20544⟩⟩) 0 ⟨14962⟩ 1975

def event44047 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20544⟩⟩) (.authority (.relationPreimageSource ⟨30⟩))

def exact44048RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20544⟩⟩]⟩, (1)⟩]

theorem exact44048RawTermsValid :
    exact44048RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44048 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20544⟩⟩) exact44048RawTerms (.finite 136065468) 44047 .exactZero (none)

def event44049 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20546⟩⟩) 0 ⟨20544⟩ 44048

def event44050 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20546⟩⟩) 1 ⟨2348⟩ 4

def event44051 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20546⟩⟩) (.scale (.predecessor 0 44049 .coefficient) (.value (.predecessor 1 44050 .coefficient)))

def exact44052RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20544⟩⟩]⟩, (1)⟩]

theorem exact44052RawTermsValid :
    exact44052RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44052 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20546⟩⟩) exact44052RawTerms (.finite 136065468) 44051 .exactZero (none)

def event44053 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20547⟩⟩) 0 ⟨5553⟩ 36137

def event44054 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20547⟩⟩) 1 ⟨20546⟩ 44052

def event44055 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20547⟩⟩) (.product (.predecessor 0 44053 .coefficient) (.predecessor 1 44054 .coefficient) (⟨false, false, none, none, none⟩))

def event44056 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20547⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨20544⟩⟩]⟩) [⟨.result 44048 .coefficient, false, none⟩])

def event44057 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20547⟩⟩) (.product (.result 36137 .summary) (.transfer 44056) (⟨false, false, none, none, none⟩))

def event44058 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20547⟩⟩, .operator (⟨36137, 0⟩, ⟨44052, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20544⟩⟩]⟩, (1)⟩)

def event44059 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨20545⟩⟩)

def event44060 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event44061 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event44062 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.authority (.operator))

def event44063 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.finite 4)

def event44064 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event44065 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event44066 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event44067 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event44068 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 44067

def event44069 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 44065

def event44070 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 44068 .coefficient) (.value (.predecessor 1 44069 .coefficient)))

def event44071 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event44072 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 0 ⟨5503⟩ 44071

def event44073 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 1 ⟨4733⟩ 44063

def event44074 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.sum [.predecessor 0 44072 .coefficient, .predecessor 1 44073 .coefficient])

def event44075 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.finite 221)

def event44076 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 0 ⟨5512⟩ 44075

def event44077 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 1 ⟨961⟩ 44061

def event44078 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.identity (.predecessor 1 44077 .coefficient))

def event44079 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.finite 224)

def event44080 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10692⟩⟩) 0 ⟨5548⟩ 44079

def event44081 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10692⟩⟩) (.authority (.programFamilyFact))

def exact44082RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10692⟩⟩], []⟩, (1)⟩]

theorem exact44082RawTermsValid :
    exact44082RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44082 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10692⟩⟩) exact44082RawTerms (.finite 3) 44081 .exactZero (none)

def event44083 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9515⟩⟩) 0 ⟨5548⟩ 44079

def event44084 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9515⟩⟩) (.authority (.programFamilyFact))

def exact44085RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9515⟩⟩], []⟩, (1)⟩]

theorem exact44085RawTermsValid :
    exact44085RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44085 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9515⟩⟩) exact44085RawTerms (.finite 3) 44084 .exactZero (none)

def event44086 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10693⟩⟩) 0 ⟨9515⟩ 44085

def event44087 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10693⟩⟩) 1 ⟨10692⟩ 44082

def event44088 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10693⟩⟩) (.product (.predecessor 0 44086 .coefficient) (.predecessor 1 44087 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event44089 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10693⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], []⟩) [⟨.result 44085 .coefficient, true, some 1⟩, ⟨.result 44082 .coefficient, true, some 1⟩])

def event44090 : Event := .survivorFold (1) 44089

def exact44091RawTerms : List Term := []

theorem exact44091RawTermsValid :
    exact44091RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44091 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10693⟩⟩) exact44091RawTerms (.finite 9) 44088 (.finite 9) (some (44089))

def event44092 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10694⟩⟩) 0 ⟨10693⟩ 44091

def event44093 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10694⟩⟩) (.identity (.predecessor 0 44092 .coefficient))

def event44094 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10694⟩⟩) (.finite 9)

def event44095 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14961⟩⟩) 0 ⟨10694⟩ 44094

def event44096 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14961⟩⟩) (.authority (.programFamilyFact))

def exact44097RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14961⟩⟩], []⟩, (1)⟩]

theorem exact44097RawTermsValid :
    exact44097RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44097 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14961⟩⟩) exact44097RawTerms (.finite 3) 44096 .exactZero (none)

def event44098 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14962⟩⟩) 0 ⟨14961⟩ 44097

def event44099 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14962⟩⟩) (.identity (.predecessor 0 44098 .coefficient))

def event44100 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14962⟩⟩) (.finite 3)

def event44101 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20544⟩⟩) 0 ⟨14962⟩ 44100

def event44102 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20544⟩⟩) (.authority (.relationPreimageSource ⟨30⟩))

def exact44103RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨20544⟩⟩]⟩, (1)⟩]

theorem exact44103RawTermsValid :
    exact44103RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44103 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20544⟩⟩) exact44103RawTerms (.finite 136065468) 44102 .exactZero (none)

def event44104 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact44105RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact44105RawTermsValid :
    exact44105RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44105 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact44105RawTerms .large 44104 .exactZero (none)

def event44106 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20545⟩⟩) 0 ⟨6⟩ 44105

def event44107 : Event := .predecessor (⟨.program ⟨214⟩, ⟨20545⟩⟩) 1 ⟨20544⟩ 44103

def event44108 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨20545⟩⟩) (.product (.predecessor 0 44106 .coefficient) (.predecessor 1 44107 .coefficient) (⟨false, false, none, none, none⟩))

def event44109 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20545⟩⟩, .operator (⟨44105, 0⟩, ⟨44103, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20544⟩⟩]⟩, (1)⟩)

def exact44110RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20544⟩⟩]⟩, (1)⟩]

theorem exact44110RawTermsValid :
    exact44110RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44110 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20545⟩⟩) exact44110RawTerms .large 44108 .exactZero (none)

def event44111 : Event := .preFoldPolynomial 44110 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20544⟩⟩]⟩, (1)⟩] .exactZero none

def exact44112RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20544⟩⟩]⟩, (1)⟩]

def event44112 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨20545⟩⟩) 44111 exact44112RawTerms .large 44108 .exactZero (none)

def event44113 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨26595⟩⟩)

def event44114 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event44115 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event44116 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.authority (.operator))

def event44117 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨4733⟩⟩) (.finite 4)

def event44118 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event44119 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event44120 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event44121 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event44122 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 44121

def event44123 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 44119

def event44124 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 44122 .coefficient) (.value (.predecessor 1 44123 .coefficient)))

def event44125 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event44126 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 0 ⟨5503⟩ 44125

def event44127 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5512⟩⟩) 1 ⟨4733⟩ 44117

def event44128 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.sum [.predecessor 0 44126 .coefficient, .predecessor 1 44127 .coefficient])

def event44129 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5512⟩⟩) (.finite 221)

def event44130 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 0 ⟨5512⟩ 44129

def event44131 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5548⟩⟩) 1 ⟨961⟩ 44115

def event44132 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.identity (.predecessor 1 44131 .coefficient))

def event44133 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5548⟩⟩) (.finite 224)

def event44134 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10692⟩⟩) 0 ⟨5548⟩ 44133

def event44135 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10692⟩⟩) (.authority (.programFamilyFact))

def exact44136RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨10692⟩⟩], []⟩, (1)⟩]

theorem exact44136RawTermsValid :
    exact44136RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44136 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10692⟩⟩) exact44136RawTerms (.finite 3) 44135 .exactZero (none)

def event44137 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9515⟩⟩) 0 ⟨5548⟩ 44133

def event44138 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9515⟩⟩) (.authority (.programFamilyFact))

def exact44139RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9515⟩⟩], []⟩, (1)⟩]

theorem exact44139RawTermsValid :
    exact44139RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44139 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9515⟩⟩) exact44139RawTerms (.finite 3) 44138 .exactZero (none)

def event44140 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10693⟩⟩) 0 ⟨9515⟩ 44139

def event44141 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10693⟩⟩) 1 ⟨10692⟩ 44136

def event44142 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10693⟩⟩) (.product (.predecessor 0 44140 .coefficient) (.predecessor 1 44141 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event44143 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10693⟩⟩, .operator (⟨44139, 0⟩, ⟨44136, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], []⟩, (1)⟩)

def exact44144RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], []⟩, (1)⟩]

theorem exact44144RawTermsValid :
    exact44144RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44144 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10693⟩⟩) exact44144RawTerms (.finite 9) 44142 .exactZero (none)

def event44145 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10694⟩⟩) 0 ⟨10693⟩ 44144

def event44146 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10694⟩⟩) (.identity (.predecessor 0 44145 .coefficient))

def event44147 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨10694⟩⟩) (.finite 9)

def event44148 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14961⟩⟩) 0 ⟨10694⟩ 44147

def event44149 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14961⟩⟩) (.authority (.programFamilyFact))

def exact44150RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14961⟩⟩], []⟩, (1)⟩]

theorem exact44150RawTermsValid :
    exact44150RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44150 : Event := .resultExact (⟨.program ⟨214⟩, ⟨14961⟩⟩) exact44150RawTerms (.finite 3) 44149 .exactZero (none)

def event44151 : Event := .predecessor (⟨.program ⟨214⟩, ⟨14962⟩⟩) 0 ⟨14961⟩ 44150

def event44152 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨14962⟩⟩) (.identity (.predecessor 0 44151 .coefficient))

def event44153 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨14962⟩⟩) (.finite 3)

def event44154 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23788⟩⟩) 0 ⟨14962⟩ 44153

def event44155 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23788⟩⟩) (.authority (.programFamilyFact))

def event44156 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23788⟩⟩) (.finite 3720)

def event44157 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event44158 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23790⟩⟩) 0 ⟨6689⟩ 44157

def event44159 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23790⟩⟩) 1 ⟨23788⟩ 44156

def event44160 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23790⟩⟩) (.authority (.operator))

def exact44161RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23790⟩⟩]⟩, (1)⟩]

theorem exact44161RawTermsValid :
    exact44161RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44161 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23790⟩⟩) exact44161RawTerms .large 44160 .exactZero (none)

def event44162 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26590⟩⟩) 0 ⟨23790⟩ 44161

def event44163 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26590⟩⟩) (.authority (.operator))

def exact44164RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26590⟩⟩]⟩, (1)⟩]

theorem exact44164RawTermsValid :
    exact44164RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44164 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26590⟩⟩) exact44164RawTerms (.finite 8192) 44163 .exactZero (none)

def event44165 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event44166 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event44167 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15001⟩⟩) 0 ⟨14962⟩ 44153

def event44168 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15001⟩⟩) 1 ⟨110⟩ 44166

def event44169 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15001⟩⟩) (.sum [.predecessor 0 44167 .coefficient, .predecessor 1 44168 .coefficient])

def event44170 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨15001⟩⟩) (.finite 3)

def event44171 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15002⟩⟩) 0 ⟨15001⟩ 44170

def event44172 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15002⟩⟩) (.identity (.predecessor 0 44171 .coefficient))

def exact44173RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14961⟩⟩], []⟩, (1)⟩]

theorem exact44173RawTermsValid :
    exact44173RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44173 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15002⟩⟩) exact44173RawTerms (.finite 3) 44172 .exactZero (none)

def event44174 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact44175RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact44175RawTermsValid :
    exact44175RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44175 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact44175RawTerms .large 44174 .exactZero (none)

def event44176 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15003⟩⟩) 0 ⟨6544⟩ 44175

def event44177 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15003⟩⟩) 1 ⟨15002⟩ 44173

def event44178 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15003⟩⟩) (.product (.predecessor 0 44176 .coefficient) (.predecessor 1 44177 .coefficient) (⟨false, false, none, none, none⟩))

def event44179 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15003⟩⟩, .operator (⟨44175, 0⟩, ⟨44173, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact44180RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact44180RawTermsValid :
    exact44180RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44180 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15003⟩⟩) exact44180RawTerms .large 44178 .exactZero (none)

def event44181 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6691⟩⟩) 0 ⟨6689⟩ 44157

def event44182 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6691⟩⟩) (.authority (.operator))

def exact44183RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩]

theorem exact44183RawTermsValid :
    exact44183RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44183 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6691⟩⟩) exact44183RawTerms .large 44182 .exactZero (none)

def event44184 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15004⟩⟩) 0 ⟨6691⟩ 44183

def event44185 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15004⟩⟩) 1 ⟨15003⟩ 44180

def event44186 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15004⟩⟩) (.sum [.predecessor 0 44184 .coefficient, .predecessor 1 44185 .coefficient])

def exact44187RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44187RawTermsValid :
    exact44187RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44187 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15004⟩⟩) exact44187RawTerms .large 44186 .exactZero (none)

def event44188 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26591⟩⟩) 0 ⟨15004⟩ 44187

def event44189 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26591⟩⟩) 1 ⟨26590⟩ 44164

def event44190 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26591⟩⟩) (.product (.predecessor 0 44188 .coefficient) (.predecessor 1 44189 .coefficient) (⟨false, false, none, none, none⟩))

def event44191 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26591⟩⟩, .operator (⟨44187, 0⟩, ⟨44164, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26590⟩⟩]⟩, (1)⟩)

def event44192 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26591⟩⟩, .operator (⟨44187, 1⟩, ⟨44164, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26590⟩⟩]⟩, (-1)⟩)

def event44193 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨26591⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26590⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26590⟩⟩) ⟨23790⟩ 44161)

def event44194 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26591⟩⟩, .relation 44193 0, ⟨[⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨23790⟩⟩]⟩, (-1)⟩)

def exact44195RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26590⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨23790⟩⟩]⟩, (-1)⟩]

theorem exact44195RawTermsValid :
    exact44195RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44195 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26591⟩⟩) exact44195RawTerms .large 44190 .exactZero (none)

def event44196 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15318⟩⟩) 0 ⟨14962⟩ 44153

def event44197 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15318⟩⟩) (.authority (.programFamilyFact))

def exact44198RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15318⟩⟩], []⟩, (1)⟩]

theorem exact44198RawTermsValid :
    exact44198RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44198 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15318⟩⟩) exact44198RawTerms (.finite 48) 44197 .exactZero (none)

def event44199 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15320⟩⟩) 0 ⟨6544⟩ 44175

def event44200 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15320⟩⟩) 1 ⟨15318⟩ 44198

def event44201 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15320⟩⟩) (.product (.predecessor 0 44199 .coefficient) (.predecessor 1 44200 .coefficient) (⟨false, true, none, none, some 1⟩))

def event44202 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨15320⟩⟩, .operator (⟨44175, 0⟩, ⟨44198, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact44203RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact44203RawTermsValid :
    exact44203RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44203 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15320⟩⟩) exact44203RawTerms .large 44201 .exactZero (none)

def event44204 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6711⟩⟩) 0 ⟨6689⟩ 44157

def event44205 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6711⟩⟩) (.authority (.operator))

def exact44206RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩]

theorem exact44206RawTermsValid :
    exact44206RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44206 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6711⟩⟩) exact44206RawTerms .large 44205 .exactZero (none)

def event44207 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15321⟩⟩) 0 ⟨6711⟩ 44206

def event44208 : Event := .predecessor (⟨.program ⟨214⟩, ⟨15321⟩⟩) 1 ⟨15320⟩ 44203

def event44209 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨15321⟩⟩) (.sum [.predecessor 0 44207 .coefficient, .predecessor 1 44208 .coefficient])

def exact44210RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44210RawTermsValid :
    exact44210RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44210 : Event := .resultExact (⟨.program ⟨214⟩, ⟨15321⟩⟩) exact44210RawTerms .large 44209 .exactZero (none)

def event44211 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26595⟩⟩) 0 ⟨15321⟩ 44210

def event44212 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26595⟩⟩) 1 ⟨26591⟩ 44195

def event44213 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26595⟩⟩) (.sum [.predecessor 0 44211 .coefficient, .predecessor 1 44212 .coefficient])

def exact44214RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26590⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨23790⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44214RawTermsValid :
    exact44214RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44214 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26595⟩⟩) exact44214RawTerms .large 44213 .exactZero (none)

def event44215 : Event := .preFoldPolynomial 44214 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26590⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨23790⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact44216RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26590⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨23790⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event44216 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨26595⟩⟩) 44215 exact44216RawTerms .large 44213 .exactZero (none)

def event44217 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨14962⟩⟩) ⟨⟨124⟩, ⟨30⟩, ⟨109⟩⟩ ⟨44059, 44217⟩

def event44218 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨20547⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20544⟩⟩]⟩) (1) 0 2 (.universal 44217 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20544⟩⟩]⟩) (none) 44216)

def event44219 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20547⟩⟩, .relation 44218 1, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩)

def event44220 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20547⟩⟩, .relation 44218 0, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26590⟩⟩]⟩, (-1)⟩)

def event44221 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20547⟩⟩, .relation 44218 2, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨23790⟩⟩]⟩, (1)⟩)

def event44222 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨20547⟩⟩, .relation 44218 3, ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact44223RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26590⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨23790⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44223RawTermsValid :
    exact44223RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44223 : Event := .resultExact (⟨.program ⟨214⟩, ⟨20547⟩⟩) exact44223RawTerms .large 44055 (.finite 1811303510016) (some (44057))

def event44224 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26593⟩⟩) 0 ⟨20547⟩ 44223

def event44225 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26593⟩⟩) 1 ⟨26592⟩ 44045

def event44226 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26593⟩⟩) (.sum [.predecessor 0 44224 .coefficient, .predecessor 1 44225 .coefficient])

def event44227 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26593⟩⟩, .operator (⟨44223, 0⟩, ⟨44045, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26590⟩⟩]⟩, (1)⟩)

def event44228 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨26593⟩⟩, .operator (⟨44223, 2⟩, ⟨44045, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨23790⟩⟩]⟩, (-1)⟩)

def event44229 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26593⟩⟩) (.sum [.result 44223 .summary, .result 44045 .summary])

def exact44230RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44230RawTermsValid :
    exact44230RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44230 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26593⟩⟩) exact44230RawTerms .large 44226 (.finite 1291900380601931935744) (some (44229))

def event44231 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23725⟩⟩) 0 ⟨14801⟩ 1998

def event44232 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23725⟩⟩) (.authority (.programFamilyFact))

def event44233 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23725⟩⟩) (.finite 3720)

def event44234 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23727⟩⟩) 0 ⟨6689⟩ 5477

def event44235 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23727⟩⟩) 1 ⟨23725⟩ 44233

def event44236 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23727⟩⟩) (.authority (.operator))

def exact44237RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23727⟩⟩]⟩, (1)⟩]

theorem exact44237RawTermsValid :
    exact44237RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44237 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23727⟩⟩) exact44237RawTerms .large 44236 .exactZero (none)

def event44238 : Event := .predecessor (⟨.program ⟨214⟩, ⟨26382⟩⟩) 0 ⟨23727⟩ 44237

def event44239 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨26382⟩⟩) (.authority (.operator))

def exact44240RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨26382⟩⟩]⟩, (1)⟩]

theorem exact44240RawTermsValid :
    exact44240RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44240 : Event := .resultExact (⟨.program ⟨214⟩, ⟨26382⟩⟩) exact44240RawTerms (.finite 8192) 44239 .exactZero (none)

def event44241 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22957⟩⟩) 0 ⟨10498⟩ 1992

def event44242 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22957⟩⟩) (.authority (.programFamilyFact))

def event44243 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨22957⟩⟩) (.finite 3720)

def event44244 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22958⟩⟩) 0 ⟨6689⟩ 5477

def event44245 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22958⟩⟩) 1 ⟨22957⟩ 44243

def event44246 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22958⟩⟩) (.authority (.operator))

def exact44247RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22958⟩⟩]⟩, (1)⟩]

theorem exact44247RawTermsValid :
    exact44247RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44247 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22958⟩⟩) exact44247RawTerms .large 44246 .exactZero (none)

def event44248 : Event := .predecessor (⟨.program ⟨214⟩, ⟨24921⟩⟩) 0 ⟨22958⟩ 44247

def event44249 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨24921⟩⟩) (.authority (.operator))

def exact44250RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨24921⟩⟩]⟩, (1)⟩]

theorem exact44250RawTermsValid :
    exact44250RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44250 : Event := .resultExact (⟨.program ⟨214⟩, ⟨24921⟩⟩) exact44250RawTerms (.finite 8192) 44249 .exactZero (none)

def event44251 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10499⟩⟩) 0 ⟨10496⟩ 1981

def event44252 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10499⟩⟩) 1 ⟨6569⟩ 36045

def event44253 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10499⟩⟩) (.tensor (.predecessor 0 44251 .coefficient) (.predecessor 1 44252 .coefficient) true false)

def event44254 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10499⟩⟩, .operator (⟨1981, 0⟩, ⟨36045, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact44255RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact44255RawTermsValid :
    exact44255RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44255 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10499⟩⟩) exact44255RawTerms .large 44253 .exactZero (none)

def event44256 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7304⟩⟩) 0 ⟨5551⟩ 35915

def event44257 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7304⟩⟩) 1 ⟨6772⟩ 14989

def event44258 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7304⟩⟩) (.product (.predecessor 0 44256 .coefficient) (.predecessor 1 44257 .coefficient) (⟨false, false, none, none, none⟩))

def event44259 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7304⟩⟩, .operator (⟨35915, 0⟩, ⟨14989, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩)

def exact44260RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩]

theorem exact44260RawTermsValid :
    exact44260RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44260 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7304⟩⟩) exact44260RawTerms .large 44258 .exactZero (none)

def event44261 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10500⟩⟩) 0 ⟨7304⟩ 44260

def event44262 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10500⟩⟩) 1 ⟨10499⟩ 44255

def event44263 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10500⟩⟩) (.sum [.predecessor 0 44261 .coefficient, .predecessor 1 44262 .coefficient])

def exact44264RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44264RawTermsValid :
    exact44264RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44264 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10500⟩⟩) exact44264RawTerms .large 44263 .exactZero (none)

def event44265 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10501⟩⟩) 0 ⟨10500⟩ 44264

def event44266 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10501⟩⟩) 1 ⟨86⟩ 14981

def event44267 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10501⟩⟩) (.sum [.predecessor 0 44265 .coefficient, .predecessor 1 44266 .coefficient])

def event44268 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10501⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨86⟩⟩]⟩) [⟨.result 14981 .coefficient, false, none⟩])

def event44269 : Event := .survivorFold (1) 44268

def exact44270RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44270RawTermsValid :
    exact44270RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44270 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10501⟩⟩) exact44270RawTerms .large 44267 (.finite 26) (some (44268))

def event44271 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10502⟩⟩) 0 ⟨10501⟩ 44270

def event44272 : Event := .predecessor (⟨.program ⟨214⟩, ⟨10502⟩⟩) 1 ⟨9410⟩ 1984

def event44273 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10502⟩⟩) (.product (.predecessor 0 44271 .coefficient) (.predecessor 1 44272 .coefficient) (⟨false, true, none, none, some 1⟩))

def event44274 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10502⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9410⟩⟩], []⟩) [⟨.result 1984 .coefficient, true, some 1⟩])

def event44275 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨10502⟩⟩) (.product (.result 44270 .summary) (.transfer 44274) (⟨false, false, none, none, none⟩))

def event44276 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10502⟩⟩, .operator (⟨44270, 1⟩, ⟨1984, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def event44277 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨10502⟩⟩, .operator (⟨44270, 0⟩, ⟨1984, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩)

def exact44278RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact44278RawTermsValid :
    exact44278RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44278 : Event := .resultExact (⟨.program ⟨214⟩, ⟨10502⟩⟩) exact44278RawTerms .large 44273 (.finite 1664) (some (44275))

def event44279 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9411⟩⟩) 0 ⟨9410⟩ 1984

def event44280 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9411⟩⟩) 1 ⟨6569⟩ 36045

def event44281 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9411⟩⟩) (.tensor (.predecessor 0 44279 .coefficient) (.predecessor 1 44280 .coefficient) true false)

def event44282 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨9411⟩⟩, .operator (⟨1984, 0⟩, ⟨36045, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact44283RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact44283RawTermsValid :
    exact44283RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event44283 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9411⟩⟩) exact44283RawTerms .large 44281 .exactZero (none)

def event44284 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7303⟩⟩) 0 ⟨5551⟩ 35915

def event44285 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7303⟩⟩) 1 ⟨6771⟩ 15030

def event44286 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7303⟩⟩) (.product (.predecessor 0 44284 .coefficient) (.predecessor 1 44285 .coefficient) (⟨false, false, none, none, none⟩))

def event44287 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7303⟩⟩, .operator (⟨35915, 0⟩, ⟨15030, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩]⟩, (1)⟩)

def eventLeaf2752 : Array AnnotatedEvent := #[
  { event := event44032
    frameStart := 0 },
  { event := event44033
    frameStart := 0 },
  { event := event44034
    frameStart := 0 },
  { event := event44035
    frameStart := 0 },
  { event := event44036
    frameStart := 0 },
  { event := event44037
    frameStart := 0 },
  { event := event44038
    frameStart := 0 },
  { event := event44039
    frameStart := 0 },
  { event := event44040
    frameStart := 0 },
  { event := event44041
    frameStart := 0 },
  { event := event44042
    frameStart := 0 },
  { event := event44043
    frameStart := 0 },
  { event := event44044
    frameStart := 0 },
  { event := event44045
    frameStart := 0 },
  { event := event44046
    frameStart := 0 },
  { event := event44047
    frameStart := 0 }
]

def eventLeaf2753 : Array AnnotatedEvent := #[
  { event := event44048
    frameStart := 0 },
  { event := event44049
    frameStart := 0 },
  { event := event44050
    frameStart := 0 },
  { event := event44051
    frameStart := 0 },
  { event := event44052
    frameStart := 0 },
  { event := event44053
    frameStart := 0 },
  { event := event44054
    frameStart := 0 },
  { event := event44055
    frameStart := 0 },
  { event := event44056
    frameStart := 0 },
  { event := event44057
    frameStart := 0 },
  { event := event44058
    frameStart := 0 },
  { event := event44059
    frameStart := 44059 },
  { event := event44060
    frameStart := 44059 },
  { event := event44061
    frameStart := 44059 },
  { event := event44062
    frameStart := 44059 },
  { event := event44063
    frameStart := 44059 }
]

def eventLeaf2754 : Array AnnotatedEvent := #[
  { event := event44064
    frameStart := 44059 },
  { event := event44065
    frameStart := 44059 },
  { event := event44066
    frameStart := 44059 },
  { event := event44067
    frameStart := 44059 },
  { event := event44068
    frameStart := 44059 },
  { event := event44069
    frameStart := 44059 },
  { event := event44070
    frameStart := 44059 },
  { event := event44071
    frameStart := 44059 },
  { event := event44072
    frameStart := 44059 },
  { event := event44073
    frameStart := 44059 },
  { event := event44074
    frameStart := 44059 },
  { event := event44075
    frameStart := 44059 },
  { event := event44076
    frameStart := 44059 },
  { event := event44077
    frameStart := 44059 },
  { event := event44078
    frameStart := 44059 },
  { event := event44079
    frameStart := 44059 }
]

def eventLeaf2755 : Array AnnotatedEvent := #[
  { event := event44080
    frameStart := 44059 },
  { event := event44081
    frameStart := 44059 },
  { event := event44082
    frameStart := 44059 },
  { event := event44083
    frameStart := 44059 },
  { event := event44084
    frameStart := 44059 },
  { event := event44085
    frameStart := 44059 },
  { event := event44086
    frameStart := 44059 },
  { event := event44087
    frameStart := 44059 },
  { event := event44088
    frameStart := 44059 },
  { event := event44089
    frameStart := 44059 },
  { event := event44090
    frameStart := 44059 },
  { event := event44091
    frameStart := 44059 },
  { event := event44092
    frameStart := 44059 },
  { event := event44093
    frameStart := 44059 },
  { event := event44094
    frameStart := 44059 },
  { event := event44095
    frameStart := 44059 }
]

def eventLeaf2756 : Array AnnotatedEvent := #[
  { event := event44096
    frameStart := 44059 },
  { event := event44097
    frameStart := 44059 },
  { event := event44098
    frameStart := 44059 },
  { event := event44099
    frameStart := 44059 },
  { event := event44100
    frameStart := 44059 },
  { event := event44101
    frameStart := 44059 },
  { event := event44102
    frameStart := 44059 },
  { event := event44103
    frameStart := 44059 },
  { event := event44104
    frameStart := 44059 },
  { event := event44105
    frameStart := 44059 },
  { event := event44106
    frameStart := 44059 },
  { event := event44107
    frameStart := 44059 },
  { event := event44108
    frameStart := 44059 },
  { event := event44109
    frameStart := 44059 },
  { event := event44110
    frameStart := 44059 },
  { event := event44111
    frameStart := 44059 }
]

def eventLeaf2757 : Array AnnotatedEvent := #[
  { event := event44112
    frameStart := 44059 },
  { event := event44113
    frameStart := 44113 },
  { event := event44114
    frameStart := 44113 },
  { event := event44115
    frameStart := 44113 },
  { event := event44116
    frameStart := 44113 },
  { event := event44117
    frameStart := 44113 },
  { event := event44118
    frameStart := 44113 },
  { event := event44119
    frameStart := 44113 },
  { event := event44120
    frameStart := 44113 },
  { event := event44121
    frameStart := 44113 },
  { event := event44122
    frameStart := 44113 },
  { event := event44123
    frameStart := 44113 },
  { event := event44124
    frameStart := 44113 },
  { event := event44125
    frameStart := 44113 },
  { event := event44126
    frameStart := 44113 },
  { event := event44127
    frameStart := 44113 }
]

def eventLeaf2758 : Array AnnotatedEvent := #[
  { event := event44128
    frameStart := 44113 },
  { event := event44129
    frameStart := 44113 },
  { event := event44130
    frameStart := 44113 },
  { event := event44131
    frameStart := 44113 },
  { event := event44132
    frameStart := 44113 },
  { event := event44133
    frameStart := 44113 },
  { event := event44134
    frameStart := 44113 },
  { event := event44135
    frameStart := 44113 },
  { event := event44136
    frameStart := 44113 },
  { event := event44137
    frameStart := 44113 },
  { event := event44138
    frameStart := 44113 },
  { event := event44139
    frameStart := 44113 },
  { event := event44140
    frameStart := 44113 },
  { event := event44141
    frameStart := 44113 },
  { event := event44142
    frameStart := 44113 },
  { event := event44143
    frameStart := 44113 }
]

def eventLeaf2759 : Array AnnotatedEvent := #[
  { event := event44144
    frameStart := 44113 },
  { event := event44145
    frameStart := 44113 },
  { event := event44146
    frameStart := 44113 },
  { event := event44147
    frameStart := 44113 },
  { event := event44148
    frameStart := 44113 },
  { event := event44149
    frameStart := 44113 },
  { event := event44150
    frameStart := 44113 },
  { event := event44151
    frameStart := 44113 },
  { event := event44152
    frameStart := 44113 },
  { event := event44153
    frameStart := 44113 },
  { event := event44154
    frameStart := 44113 },
  { event := event44155
    frameStart := 44113 },
  { event := event44156
    frameStart := 44113 },
  { event := event44157
    frameStart := 44113 },
  { event := event44158
    frameStart := 44113 },
  { event := event44159
    frameStart := 44113 }
]

def eventLeaf2760 : Array AnnotatedEvent := #[
  { event := event44160
    frameStart := 44113 },
  { event := event44161
    frameStart := 44113 },
  { event := event44162
    frameStart := 44113 },
  { event := event44163
    frameStart := 44113 },
  { event := event44164
    frameStart := 44113 },
  { event := event44165
    frameStart := 44113 },
  { event := event44166
    frameStart := 44113 },
  { event := event44167
    frameStart := 44113 },
  { event := event44168
    frameStart := 44113 },
  { event := event44169
    frameStart := 44113 },
  { event := event44170
    frameStart := 44113 },
  { event := event44171
    frameStart := 44113 },
  { event := event44172
    frameStart := 44113 },
  { event := event44173
    frameStart := 44113 },
  { event := event44174
    frameStart := 44113 },
  { event := event44175
    frameStart := 44113 }
]

def eventLeaf2761 : Array AnnotatedEvent := #[
  { event := event44176
    frameStart := 44113 },
  { event := event44177
    frameStart := 44113 },
  { event := event44178
    frameStart := 44113 },
  { event := event44179
    frameStart := 44113 },
  { event := event44180
    frameStart := 44113 },
  { event := event44181
    frameStart := 44113 },
  { event := event44182
    frameStart := 44113 },
  { event := event44183
    frameStart := 44113 },
  { event := event44184
    frameStart := 44113 },
  { event := event44185
    frameStart := 44113 },
  { event := event44186
    frameStart := 44113 },
  { event := event44187
    frameStart := 44113 },
  { event := event44188
    frameStart := 44113 },
  { event := event44189
    frameStart := 44113 },
  { event := event44190
    frameStart := 44113 },
  { event := event44191
    frameStart := 44113 }
]

def eventLeaf2762 : Array AnnotatedEvent := #[
  { event := event44192
    frameStart := 44113 },
  { event := event44193
    frameStart := 44113 },
  { event := event44194
    frameStart := 44113 },
  { event := event44195
    frameStart := 44113 },
  { event := event44196
    frameStart := 44113 },
  { event := event44197
    frameStart := 44113 },
  { event := event44198
    frameStart := 44113 },
  { event := event44199
    frameStart := 44113 },
  { event := event44200
    frameStart := 44113 },
  { event := event44201
    frameStart := 44113 },
  { event := event44202
    frameStart := 44113 },
  { event := event44203
    frameStart := 44113 },
  { event := event44204
    frameStart := 44113 },
  { event := event44205
    frameStart := 44113 },
  { event := event44206
    frameStart := 44113 },
  { event := event44207
    frameStart := 44113 }
]

def eventLeaf2763 : Array AnnotatedEvent := #[
  { event := event44208
    frameStart := 44113 },
  { event := event44209
    frameStart := 44113 },
  { event := event44210
    frameStart := 44113 },
  { event := event44211
    frameStart := 44113 },
  { event := event44212
    frameStart := 44113 },
  { event := event44213
    frameStart := 44113 },
  { event := event44214
    frameStart := 44113 },
  { event := event44215
    frameStart := 44113 },
  { event := event44216
    frameStart := 44113 },
  { event := event44217
    frameStart := 0 },
  { event := event44218
    frameStart := 0 },
  { event := event44219
    frameStart := 0 },
  { event := event44220
    frameStart := 0 },
  { event := event44221
    frameStart := 0 },
  { event := event44222
    frameStart := 0 },
  { event := event44223
    frameStart := 0 }
]

def eventLeaf2764 : Array AnnotatedEvent := #[
  { event := event44224
    frameStart := 0 },
  { event := event44225
    frameStart := 0 },
  { event := event44226
    frameStart := 0 },
  { event := event44227
    frameStart := 0 },
  { event := event44228
    frameStart := 0 },
  { event := event44229
    frameStart := 0 },
  { event := event44230
    frameStart := 0 },
  { event := event44231
    frameStart := 0 },
  { event := event44232
    frameStart := 0 },
  { event := event44233
    frameStart := 0 },
  { event := event44234
    frameStart := 0 },
  { event := event44235
    frameStart := 0 },
  { event := event44236
    frameStart := 0 },
  { event := event44237
    frameStart := 0 },
  { event := event44238
    frameStart := 0 },
  { event := event44239
    frameStart := 0 }
]

def eventLeaf2765 : Array AnnotatedEvent := #[
  { event := event44240
    frameStart := 0 },
  { event := event44241
    frameStart := 0 },
  { event := event44242
    frameStart := 0 },
  { event := event44243
    frameStart := 0 },
  { event := event44244
    frameStart := 0 },
  { event := event44245
    frameStart := 0 },
  { event := event44246
    frameStart := 0 },
  { event := event44247
    frameStart := 0 },
  { event := event44248
    frameStart := 0 },
  { event := event44249
    frameStart := 0 },
  { event := event44250
    frameStart := 0 },
  { event := event44251
    frameStart := 0 },
  { event := event44252
    frameStart := 0 },
  { event := event44253
    frameStart := 0 },
  { event := event44254
    frameStart := 0 },
  { event := event44255
    frameStart := 0 }
]

def eventLeaf2766 : Array AnnotatedEvent := #[
  { event := event44256
    frameStart := 0 },
  { event := event44257
    frameStart := 0 },
  { event := event44258
    frameStart := 0 },
  { event := event44259
    frameStart := 0 },
  { event := event44260
    frameStart := 0 },
  { event := event44261
    frameStart := 0 },
  { event := event44262
    frameStart := 0 },
  { event := event44263
    frameStart := 0 },
  { event := event44264
    frameStart := 0 },
  { event := event44265
    frameStart := 0 },
  { event := event44266
    frameStart := 0 },
  { event := event44267
    frameStart := 0 },
  { event := event44268
    frameStart := 0 },
  { event := event44269
    frameStart := 0 },
  { event := event44270
    frameStart := 0 },
  { event := event44271
    frameStart := 0 }
]

def eventLeaf2767 : Array AnnotatedEvent := #[
  { event := event44272
    frameStart := 0 },
  { event := event44273
    frameStart := 0 },
  { event := event44274
    frameStart := 0 },
  { event := event44275
    frameStart := 0 },
  { event := event44276
    frameStart := 0 },
  { event := event44277
    frameStart := 0 },
  { event := event44278
    frameStart := 0 },
  { event := event44279
    frameStart := 0 },
  { event := event44280
    frameStart := 0 },
  { event := event44281
    frameStart := 0 },
  { event := event44282
    frameStart := 0 },
  { event := event44283
    frameStart := 0 },
  { event := event44284
    frameStart := 0 },
  { event := event44285
    frameStart := 0 },
  { event := event44286
    frameStart := 0 },
  { event := event44287
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events172
