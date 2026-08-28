import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events375

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact96000RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25070⟩⟩], []⟩, (1)⟩]

theorem exact96000RawTermsValid :
    exact96000RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96000 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25070⟩⟩) exact96000RawTerms (.finite 16) 95999 .exactZero (none)

def event96001 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56640⟩⟩) 0 ⟨9901⟩ 95997

def event96002 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56640⟩⟩) (.authority (.programFamilyFact))

def exact96003RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56640⟩⟩], []⟩, (1)⟩]

theorem exact96003RawTermsValid :
    exact96003RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96003 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56640⟩⟩) exact96003RawTerms (.finite 16) 96002 .exactZero (none)

def event96004 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56641⟩⟩) 0 ⟨56640⟩ 96003

def event96005 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56641⟩⟩) 1 ⟨25070⟩ 96000

def event96006 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56641⟩⟩) (.product (.predecessor 0 96004 .coefficient) (.predecessor 1 96005 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event96007 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56641⟩⟩, .operator (⟨96003, 0⟩, ⟨96000, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25070⟩⟩, ⟨.program ⟨257⟩, ⟨56640⟩⟩], []⟩, (1)⟩)

def exact96008RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25070⟩⟩, ⟨.program ⟨257⟩, ⟨56640⟩⟩], []⟩, (1)⟩]

theorem exact96008RawTermsValid :
    exact96008RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96008 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56641⟩⟩) exact96008RawTerms (.finite 256) 96006 .exactZero (none)

def event96009 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56642⟩⟩) 0 ⟨56641⟩ 96008

def event96010 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56642⟩⟩) (.identity (.predecessor 0 96009 .coefficient))

def event96011 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56642⟩⟩) (.finite 256)

def event96012 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57998⟩⟩) 0 ⟨56642⟩ 96011

def event96013 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57998⟩⟩) (.authority (.programFamilyFact))

def event96014 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨57998⟩⟩) (.finite 3720)

def event96015 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event96016 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57999⟩⟩) 0 ⟨7177⟩ 96015

def event96017 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57999⟩⟩) 1 ⟨57998⟩ 96014

def event96018 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57999⟩⟩) (.authority (.operator))

def exact96019RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨57999⟩⟩]⟩, (1)⟩]

theorem exact96019RawTermsValid :
    exact96019RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96019 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57999⟩⟩) exact96019RawTerms .large 96018 .exactZero (none)

def event96020 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58534⟩⟩) 0 ⟨57999⟩ 96019

def event96021 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58534⟩⟩) (.authority (.operator))

def exact96022RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨58534⟩⟩]⟩, (1)⟩]

theorem exact96022RawTermsValid :
    exact96022RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96022 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58534⟩⟩) exact96022RawTerms (.finite 8192) 96021 .exactZero (none)

def event96023 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event96024 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event96025 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58266⟩⟩) 0 ⟨56642⟩ 96011

def event96026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58266⟩⟩) 1 ⟨136⟩ 96024

def event96027 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58266⟩⟩) (.sum [.predecessor 0 96025 .coefficient, .predecessor 1 96026 .coefficient])

def event96028 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨58266⟩⟩) (.finite 256)

def event96029 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58267⟩⟩) 0 ⟨58266⟩ 96028

def event96030 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58267⟩⟩) (.identity (.predecessor 0 96029 .coefficient))

def exact96031RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25070⟩⟩, ⟨.program ⟨257⟩, ⟨56640⟩⟩], []⟩, (1)⟩]

theorem exact96031RawTermsValid :
    exact96031RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96031 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58267⟩⟩) exact96031RawTerms (.finite 256) 96030 .exactZero (none)

def event96032 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact96033RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact96033RawTermsValid :
    exact96033RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96033 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact96033RawTerms .large 96032 .exactZero (none)

def event96034 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58268⟩⟩) 0 ⟨6908⟩ 96033

def event96035 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58268⟩⟩) 1 ⟨58267⟩ 96031

def event96036 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58268⟩⟩) (.product (.predecessor 0 96034 .coefficient) (.predecessor 1 96035 .coefficient) (⟨false, false, none, none, none⟩))

def event96037 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58268⟩⟩, .operator (⟨96033, 0⟩, ⟨96031, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25070⟩⟩, ⟨.program ⟨257⟩, ⟨56640⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact96038RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25070⟩⟩, ⟨.program ⟨257⟩, ⟨56640⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact96038RawTermsValid :
    exact96038RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96038 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58268⟩⟩) exact96038RawTerms .large 96036 .exactZero (none)

def event96039 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event96040 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event96041 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 96015

def event96042 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact96043RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact96043RawTermsValid :
    exact96043RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96043 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact96043RawTerms .large 96042 .exactZero (none)

def event96044 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7273⟩⟩) 0 ⟨7178⟩ 96043

def event96045 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7273⟩⟩) (.identity (.predecessor 0 96044 .coefficient))

def exact96046RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩, (1)⟩]

theorem exact96046RawTermsValid :
    exact96046RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96046 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7273⟩⟩) exact96046RawTerms .large 96045 .exactZero (none)

def event96047 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9532⟩⟩) 0 ⟨7273⟩ 96046

def event96048 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9532⟩⟩) (.authority (.operator))

def exact96049RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩, (1)⟩]

theorem exact96049RawTermsValid :
    exact96049RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96049 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9532⟩⟩) exact96049RawTerms (.finite 8192) 96048 .exactZero (none)

def event96050 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9533⟩⟩) 0 ⟨9532⟩ 96049

def event96051 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9533⟩⟩) 1 ⟨2370⟩ 96040

def event96052 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9533⟩⟩) (.scale (.predecessor 0 96050 .coefficient) (.value (.predecessor 1 96051 .coefficient)))

def exact96053RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩, (1)⟩]

theorem exact96053RawTermsValid :
    exact96053RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96053 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9533⟩⟩) exact96053RawTerms (.finite 8192) 96052 .exactZero (none)

def event96054 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7290⟩⟩) 0 ⟨7178⟩ 96043

def event96055 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7290⟩⟩) (.identity (.predecessor 0 96054 .coefficient))

def exact96056RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩]⟩, (1)⟩]

theorem exact96056RawTermsValid :
    exact96056RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96056 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7290⟩⟩) exact96056RawTerms .large 96055 .exactZero (none)

def event96057 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9534⟩⟩) 0 ⟨7290⟩ 96056

def event96058 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9534⟩⟩) 1 ⟨9533⟩ 96053

def event96059 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9534⟩⟩) (.product (.predecessor 0 96057 .coefficient) (.predecessor 1 96058 .coefficient) (⟨false, false, none, none, none⟩))

def event96060 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9534⟩⟩, .operator (⟨96056, 0⟩, ⟨96053, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩, (1)⟩)

def exact96061RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩, (1)⟩]

theorem exact96061RawTermsValid :
    exact96061RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96061 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9534⟩⟩) exact96061RawTerms .large 96059 .exactZero (none)

def event96062 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58269⟩⟩) 0 ⟨9534⟩ 96061

def event96063 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58269⟩⟩) 1 ⟨58268⟩ 96038

def event96064 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58269⟩⟩) (.sum [.predecessor 0 96062 .coefficient, .predecessor 1 96063 .coefficient])

def exact96065RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25070⟩⟩, ⟨.program ⟨257⟩, ⟨56640⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact96065RawTermsValid :
    exact96065RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96065 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58269⟩⟩) exact96065RawTerms .large 96064 .exactZero (none)

def event96066 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58537⟩⟩) 0 ⟨58269⟩ 96065

def event96067 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58537⟩⟩) 1 ⟨58534⟩ 96022

def event96068 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58537⟩⟩) (.product (.predecessor 0 96066 .coefficient) (.predecessor 1 96067 .coefficient) (⟨false, false, none, none, none⟩))

def event96069 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58537⟩⟩, .operator (⟨96065, 0⟩, ⟨96022, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58534⟩⟩]⟩, (1)⟩)

def event96070 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58537⟩⟩, .operator (⟨96065, 1⟩, ⟨96022, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25070⟩⟩, ⟨.program ⟨257⟩, ⟨56640⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58534⟩⟩]⟩, (-1)⟩)

def event96071 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨58537⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨25070⟩⟩, ⟨.program ⟨257⟩, ⟨56640⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58534⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58534⟩⟩) ⟨57999⟩ 96019)

def event96072 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58537⟩⟩, .relation 96071 0, ⟨[⟨.program ⟨257⟩, ⟨25070⟩⟩, ⟨.program ⟨257⟩, ⟨56640⟩⟩], [⟨.program ⟨257⟩, ⟨57999⟩⟩]⟩, (-1)⟩)

def exact96073RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58534⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25070⟩⟩, ⟨.program ⟨257⟩, ⟨56640⟩⟩], [⟨.program ⟨257⟩, ⟨57999⟩⟩]⟩, (-1)⟩]

theorem exact96073RawTermsValid :
    exact96073RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96073 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58537⟩⟩) exact96073RawTerms .large 96068 .exactZero (none)

def event96074 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56888⟩⟩) 0 ⟨56642⟩ 96011

def event96075 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56888⟩⟩) (.authority (.programFamilyFact))

def exact96076RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56888⟩⟩], []⟩, (1)⟩]

theorem exact96076RawTermsValid :
    exact96076RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96076 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56888⟩⟩) exact96076RawTerms (.finite 16) 96075 .exactZero (none)

def event96077 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56890⟩⟩) 0 ⟨6908⟩ 96033

def event96078 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56890⟩⟩) 1 ⟨56888⟩ 96076

def event96079 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56890⟩⟩) (.product (.predecessor 0 96077 .coefficient) (.predecessor 1 96078 .coefficient) (⟨false, true, none, none, some 1⟩))

def event96080 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56890⟩⟩, .operator (⟨96033, 0⟩, ⟨96076, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact96081RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact96081RawTermsValid :
    exact96081RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96081 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56890⟩⟩) exact96081RawTerms .large 96079 .exactZero (none)

def event96082 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7185⟩⟩) 0 ⟨7177⟩ 96015

def event96083 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7185⟩⟩) (.authority (.operator))

def exact96084RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩, (1)⟩]

theorem exact96084RawTermsValid :
    exact96084RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96084 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7185⟩⟩) exact96084RawTerms .large 96083 .exactZero (none)

def event96085 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56891⟩⟩) 0 ⟨7185⟩ 96084

def event96086 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56891⟩⟩) 1 ⟨56890⟩ 96081

def event96087 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56891⟩⟩) (.sum [.predecessor 0 96085 .coefficient, .predecessor 1 96086 .coefficient])

def exact96088RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact96088RawTermsValid :
    exact96088RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96088 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56891⟩⟩) exact96088RawTerms .large 96087 .exactZero (none)

def event96089 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58538⟩⟩) 0 ⟨56891⟩ 96088

def event96090 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58538⟩⟩) 1 ⟨58537⟩ 96073

def event96091 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58538⟩⟩) (.sum [.predecessor 0 96089 .coefficient, .predecessor 1 96090 .coefficient])

def exact96092RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58534⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25070⟩⟩, ⟨.program ⟨257⟩, ⟨56640⟩⟩], [⟨.program ⟨257⟩, ⟨57999⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact96092RawTermsValid :
    exact96092RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96092 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58538⟩⟩) exact96092RawTerms .large 96091 .exactZero (none)

def event96093 : Event := .preFoldPolynomial 96092 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58534⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25070⟩⟩, ⟨.program ⟨257⟩, ⟨56640⟩⟩], [⟨.program ⟨257⟩, ⟨57999⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact96094RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58534⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨25070⟩⟩, ⟨.program ⟨257⟩, ⟨56640⟩⟩], [⟨.program ⟨257⟩, ⟨57999⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event96094 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨58538⟩⟩) 96093 exact96094RawTerms .large 96091 .exactZero (none)

def event96095 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨56642⟩⟩) ⟨⟨64⟩, ⟨42⟩, ⟨135⟩⟩ ⟨95929, 96095⟩

def event96096 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨57462⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57459⟩⟩]⟩) (1) 0 2 (.universal 96095 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57459⟩⟩]⟩) (none) 96094)

def event96097 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57462⟩⟩, .relation 96096 0, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩, (1)⟩)

def event96098 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57462⟩⟩, .relation 96096 1, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58534⟩⟩]⟩, (-1)⟩)

def event96099 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57462⟩⟩, .relation 96096 2, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25070⟩⟩, ⟨.program ⟨257⟩, ⟨56640⟩⟩], [⟨.program ⟨257⟩, ⟨57999⟩⟩]⟩, (1)⟩)

def event96100 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57462⟩⟩, .relation 96096 3, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact96101RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58534⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25070⟩⟩, ⟨.program ⟨257⟩, ⟨56640⟩⟩], [⟨.program ⟨257⟩, ⟨57999⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact96101RawTermsValid :
    exact96101RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96101 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57462⟩⟩) exact96101RawTerms .large 95925 (.finite 202072841853861888) (some (95927))

def event96102 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58536⟩⟩) 0 ⟨57462⟩ 96101

def event96103 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58536⟩⟩) 1 ⟨58535⟩ 95915

def event96104 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58536⟩⟩) (.sum [.predecessor 0 96102 .coefficient, .predecessor 1 96103 .coefficient])

def event96105 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58536⟩⟩, .operator (⟨96101, 2⟩, ⟨95915, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25070⟩⟩, ⟨.program ⟨257⟩, ⟨56640⟩⟩], [⟨.program ⟨257⟩, ⟨57999⟩⟩]⟩, (-1)⟩)

def event96106 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58536⟩⟩, .operator (⟨96101, 1⟩, ⟨95915, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58534⟩⟩]⟩, (1)⟩)

def event96107 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58536⟩⟩) (.sum [.result 96101 .summary, .result 95915 .summary])

def exact96108RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact96108RawTermsValid :
    exact96108RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96108 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58536⟩⟩) exact96108RawTerms .large 96104 (.finite 2997944351807545540608) (some (96107))

def event96109 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59069⟩⟩) 0 ⟨58536⟩ 96108

def event96110 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59069⟩⟩) 1 ⟨59067⟩ 95831

def event96111 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59069⟩⟩) (.product (.predecessor 0 96109 .coefficient) (.predecessor 1 96110 .coefficient) (⟨false, false, none, none, none⟩))

def event96112 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59069⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨59067⟩⟩]⟩) [⟨.result 95831 .coefficient, false, none⟩])

def event96113 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59069⟩⟩) (.product (.result 96108 .summary) (.transfer 96112) (⟨false, false, none, none, none⟩))

def event96114 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59069⟩⟩, .operator (⟨96108, 0⟩, ⟨95831, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59067⟩⟩]⟩, (1)⟩)

def event96115 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59069⟩⟩, .operator (⟨96108, 1⟩, ⟨95831, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59067⟩⟩]⟩, (-1)⟩)

def event96116 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨59069⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59067⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨59067⟩⟩) ⟨58166⟩ 95828)

def event96117 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨59069⟩⟩, .relation 96116 0, ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨58166⟩⟩]⟩, (-1)⟩)

def exact96118RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59067⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨58166⟩⟩]⟩, (-1)⟩]

theorem exact96118RawTermsValid :
    exact96118RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96118 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59069⟩⟩) exact96118RawTerms .large 96111 (.finite 32190182365603316457354999889920) (some (96113))

def event96119 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57816⟩⟩) 0 ⟨56889⟩ 4104

def event96120 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57816⟩⟩) (.authority (.relationPreimageSource ⟨70⟩))

def exact96121RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨57816⟩⟩]⟩, (1)⟩]

theorem exact96121RawTermsValid :
    exact96121RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96121 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57816⟩⟩) exact96121RawTerms (.finite 5647228698) 96120 .exactZero (none)

def event96122 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57818⟩⟩) 0 ⟨57816⟩ 96121

def event96123 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57818⟩⟩) 1 ⟨2370⟩ 4

def event96124 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57818⟩⟩) (.scale (.predecessor 0 96122 .coefficient) (.value (.predecessor 1 96123 .coefficient)))

def exact96125RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨57816⟩⟩]⟩, (1)⟩]

theorem exact96125RawTermsValid :
    exact96125RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96125 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57818⟩⟩) exact96125RawTerms (.finite 5647228698) 96124 .exactZero (none)

def event96126 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57819⟩⟩) 0 ⟨9944⟩ 90620

def event96127 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57819⟩⟩) 1 ⟨57818⟩ 96125

def event96128 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57819⟩⟩) (.product (.predecessor 0 96126 .coefficient) (.predecessor 1 96127 .coefficient) (⟨false, false, none, none, none⟩))

def event96129 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57819⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨57816⟩⟩]⟩) [⟨.result 96121 .coefficient, false, none⟩])

def event96130 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57819⟩⟩) (.product (.result 90620 .summary) (.transfer 96129) (⟨false, false, none, none, none⟩))

def event96131 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57819⟩⟩, .operator (⟨90620, 0⟩, ⟨96125, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57816⟩⟩]⟩, (1)⟩)

def event96132 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨57817⟩⟩)

def event96133 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event96134 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event96135 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.authority (.operator))

def event96136 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.finite 14)

def event96137 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event96138 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event96139 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event96140 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event96141 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 96140

def event96142 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 96138

def event96143 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 96141 .coefficient) (.value (.predecessor 1 96142 .coefficient)))

def event96144 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event96145 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 0 ⟨392⟩ 96144

def event96146 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 1 ⟨9843⟩ 96136

def event96147 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.sum [.predecessor 0 96145 .coefficient, .predecessor 1 96146 .coefficient])

def event96148 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.finite 655354)

def event96149 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 0 ⟨9845⟩ 96148

def event96150 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 1 ⟨5426⟩ 96134

def event96151 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.identity (.predecessor 1 96150 .coefficient))

def event96152 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.finite 655360)

def event96153 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25070⟩⟩) 0 ⟨9901⟩ 96152

def event96154 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25070⟩⟩) (.authority (.programFamilyFact))

def exact96155RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25070⟩⟩], []⟩, (1)⟩]

theorem exact96155RawTermsValid :
    exact96155RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96155 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25070⟩⟩) exact96155RawTerms (.finite 16) 96154 .exactZero (none)

def event96156 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56640⟩⟩) 0 ⟨9901⟩ 96152

def event96157 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56640⟩⟩) (.authority (.programFamilyFact))

def exact96158RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56640⟩⟩], []⟩, (1)⟩]

theorem exact96158RawTermsValid :
    exact96158RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96158 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56640⟩⟩) exact96158RawTerms (.finite 16) 96157 .exactZero (none)

def event96159 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56641⟩⟩) 0 ⟨56640⟩ 96158

def event96160 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56641⟩⟩) 1 ⟨25070⟩ 96155

def event96161 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56641⟩⟩) (.product (.predecessor 0 96159 .coefficient) (.predecessor 1 96160 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event96162 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56641⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨25070⟩⟩, ⟨.program ⟨257⟩, ⟨56640⟩⟩], []⟩) [⟨.result 96158 .coefficient, true, some 1⟩, ⟨.result 96155 .coefficient, true, some 1⟩])

def event96163 : Event := .survivorFold (1) 96162

def exact96164RawTerms : List Term := []

theorem exact96164RawTermsValid :
    exact96164RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96164 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56641⟩⟩) exact96164RawTerms (.finite 256) 96161 (.finite 256) (some (96162))

def event96165 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56642⟩⟩) 0 ⟨56641⟩ 96164

def event96166 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56642⟩⟩) (.identity (.predecessor 0 96165 .coefficient))

def event96167 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56642⟩⟩) (.finite 256)

def event96168 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56888⟩⟩) 0 ⟨56642⟩ 96167

def event96169 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56888⟩⟩) (.authority (.programFamilyFact))

def exact96170RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56888⟩⟩], []⟩, (1)⟩]

theorem exact96170RawTermsValid :
    exact96170RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96170 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56888⟩⟩) exact96170RawTerms (.finite 16) 96169 .exactZero (none)

def event96171 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56889⟩⟩) 0 ⟨56888⟩ 96170

def event96172 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56889⟩⟩) (.identity (.predecessor 0 96171 .coefficient))

def event96173 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56889⟩⟩) (.finite 16)

def event96174 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57816⟩⟩) 0 ⟨56889⟩ 96173

def event96175 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57816⟩⟩) (.authority (.relationPreimageSource ⟨70⟩))

def exact96176RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨57816⟩⟩]⟩, (1)⟩]

theorem exact96176RawTermsValid :
    exact96176RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96176 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57816⟩⟩) exact96176RawTerms (.finite 5647228698) 96175 .exactZero (none)

def event96177 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact96178RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact96178RawTermsValid :
    exact96178RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96178 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact96178RawTerms .large 96177 .exactZero (none)

def event96179 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57817⟩⟩) 0 ⟨35⟩ 96178

def event96180 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57817⟩⟩) 1 ⟨57816⟩ 96176

def event96181 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57817⟩⟩) (.product (.predecessor 0 96179 .coefficient) (.predecessor 1 96180 .coefficient) (⟨false, false, none, none, none⟩))

def event96182 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57817⟩⟩, .operator (⟨96178, 0⟩, ⟨96176, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57816⟩⟩]⟩, (1)⟩)

def exact96183RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57816⟩⟩]⟩, (1)⟩]

theorem exact96183RawTermsValid :
    exact96183RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96183 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57817⟩⟩) exact96183RawTerms .large 96181 .exactZero (none)

def event96184 : Event := .preFoldPolynomial 96183 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57816⟩⟩]⟩, (1)⟩] .exactZero none

def exact96185RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57816⟩⟩]⟩, (1)⟩]

def event96185 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨57817⟩⟩) 96184 exact96185RawTerms .large 96181 .exactZero (none)

def event96186 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨59072⟩⟩)

def event96187 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event96188 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event96189 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.authority (.operator))

def event96190 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.finite 14)

def event96191 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event96192 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event96193 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event96194 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event96195 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 96194

def event96196 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 96192

def event96197 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 96195 .coefficient) (.value (.predecessor 1 96196 .coefficient)))

def event96198 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event96199 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 0 ⟨392⟩ 96198

def event96200 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 1 ⟨9843⟩ 96190

def event96201 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.sum [.predecessor 0 96199 .coefficient, .predecessor 1 96200 .coefficient])

def event96202 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.finite 655354)

def event96203 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 0 ⟨9845⟩ 96202

def event96204 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 1 ⟨5426⟩ 96188

def event96205 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.identity (.predecessor 1 96204 .coefficient))

def event96206 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.finite 655360)

def event96207 : Event := .predecessor (⟨.program ⟨257⟩, ⟨25070⟩⟩) 0 ⟨9901⟩ 96206

def event96208 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨25070⟩⟩) (.authority (.programFamilyFact))

def exact96209RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25070⟩⟩], []⟩, (1)⟩]

theorem exact96209RawTermsValid :
    exact96209RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96209 : Event := .resultExact (⟨.program ⟨257⟩, ⟨25070⟩⟩) exact96209RawTerms (.finite 16) 96208 .exactZero (none)

def event96210 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56640⟩⟩) 0 ⟨9901⟩ 96206

def event96211 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56640⟩⟩) (.authority (.programFamilyFact))

def exact96212RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56640⟩⟩], []⟩, (1)⟩]

theorem exact96212RawTermsValid :
    exact96212RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96212 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56640⟩⟩) exact96212RawTerms (.finite 16) 96211 .exactZero (none)

def event96213 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56641⟩⟩) 0 ⟨56640⟩ 96212

def event96214 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56641⟩⟩) 1 ⟨25070⟩ 96209

def event96215 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56641⟩⟩) (.product (.predecessor 0 96213 .coefficient) (.predecessor 1 96214 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event96216 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨56641⟩⟩, .operator (⟨96212, 0⟩, ⟨96209, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨25070⟩⟩, ⟨.program ⟨257⟩, ⟨56640⟩⟩], []⟩, (1)⟩)

def exact96217RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨25070⟩⟩, ⟨.program ⟨257⟩, ⟨56640⟩⟩], []⟩, (1)⟩]

theorem exact96217RawTermsValid :
    exact96217RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96217 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56641⟩⟩) exact96217RawTerms (.finite 256) 96215 .exactZero (none)

def event96218 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56642⟩⟩) 0 ⟨56641⟩ 96217

def event96219 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56642⟩⟩) (.identity (.predecessor 0 96218 .coefficient))

def event96220 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56642⟩⟩) (.finite 256)

def event96221 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56888⟩⟩) 0 ⟨56642⟩ 96220

def event96222 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56888⟩⟩) (.authority (.programFamilyFact))

def exact96223RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56888⟩⟩], []⟩, (1)⟩]

theorem exact96223RawTermsValid :
    exact96223RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96223 : Event := .resultExact (⟨.program ⟨257⟩, ⟨56888⟩⟩) exact96223RawTerms (.finite 16) 96222 .exactZero (none)

def event96224 : Event := .predecessor (⟨.program ⟨257⟩, ⟨56889⟩⟩) 0 ⟨56888⟩ 96223

def event96225 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨56889⟩⟩) (.identity (.predecessor 0 96224 .coefficient))

def event96226 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨56889⟩⟩) (.finite 16)

def event96227 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58164⟩⟩) 0 ⟨56889⟩ 96226

def event96228 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58164⟩⟩) (.authority (.programFamilyFact))

def event96229 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨58164⟩⟩) (.finite 3720)

def event96230 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event96231 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58166⟩⟩) 0 ⟨7177⟩ 96230

def event96232 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58166⟩⟩) 1 ⟨58164⟩ 96229

def event96233 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58166⟩⟩) (.authority (.operator))

def exact96234RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨58166⟩⟩]⟩, (1)⟩]

theorem exact96234RawTermsValid :
    exact96234RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96234 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58166⟩⟩) exact96234RawTerms .large 96233 .exactZero (none)

def event96235 : Event := .predecessor (⟨.program ⟨257⟩, ⟨59067⟩⟩) 0 ⟨58166⟩ 96234

def event96236 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨59067⟩⟩) (.authority (.operator))

def exact96237RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨59067⟩⟩]⟩, (1)⟩]

theorem exact96237RawTermsValid :
    exact96237RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96237 : Event := .resultExact (⟨.program ⟨257⟩, ⟨59067⟩⟩) exact96237RawTerms (.finite 8192) 96236 .exactZero (none)

def event96238 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event96239 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event96240 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58346⟩⟩) 0 ⟨56889⟩ 96226

def event96241 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58346⟩⟩) 1 ⟨136⟩ 96239

def event96242 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58346⟩⟩) (.sum [.predecessor 0 96240 .coefficient, .predecessor 1 96241 .coefficient])

def event96243 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨58346⟩⟩) (.finite 16)

def event96244 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58347⟩⟩) 0 ⟨58346⟩ 96243

def event96245 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58347⟩⟩) (.identity (.predecessor 0 96244 .coefficient))

def exact96246RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56888⟩⟩], []⟩, (1)⟩]

theorem exact96246RawTermsValid :
    exact96246RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96246 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58347⟩⟩) exact96246RawTerms (.finite 16) 96245 .exactZero (none)

def event96247 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact96248RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact96248RawTermsValid :
    exact96248RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96248 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact96248RawTerms .large 96247 .exactZero (none)

def event96249 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58348⟩⟩) 0 ⟨6908⟩ 96248

def event96250 : Event := .predecessor (⟨.program ⟨257⟩, ⟨58348⟩⟩) 1 ⟨58347⟩ 96246

def event96251 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨58348⟩⟩) (.product (.predecessor 0 96249 .coefficient) (.predecessor 1 96250 .coefficient) (⟨false, false, none, none, none⟩))

def event96252 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨58348⟩⟩, .operator (⟨96248, 0⟩, ⟨96246, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact96253RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨56888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact96253RawTermsValid :
    exact96253RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event96253 : Event := .resultExact (⟨.program ⟨257⟩, ⟨58348⟩⟩) exact96253RawTerms .large 96251 .exactZero (none)

def event96254 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7185⟩⟩) 0 ⟨7177⟩ 96230

def event96255 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7185⟩⟩) (.authority (.operator))

def eventLeaf6000 : Array AnnotatedEvent := #[
  { event := event96000
    frameStart := 95977 },
  { event := event96001
    frameStart := 95977 },
  { event := event96002
    frameStart := 95977 },
  { event := event96003
    frameStart := 95977 },
  { event := event96004
    frameStart := 95977 },
  { event := event96005
    frameStart := 95977 },
  { event := event96006
    frameStart := 95977 },
  { event := event96007
    frameStart := 95977 },
  { event := event96008
    frameStart := 95977 },
  { event := event96009
    frameStart := 95977 },
  { event := event96010
    frameStart := 95977 },
  { event := event96011
    frameStart := 95977 },
  { event := event96012
    frameStart := 95977 },
  { event := event96013
    frameStart := 95977 },
  { event := event96014
    frameStart := 95977 },
  { event := event96015
    frameStart := 95977 }
]

def eventLeaf6001 : Array AnnotatedEvent := #[
  { event := event96016
    frameStart := 95977 },
  { event := event96017
    frameStart := 95977 },
  { event := event96018
    frameStart := 95977 },
  { event := event96019
    frameStart := 95977 },
  { event := event96020
    frameStart := 95977 },
  { event := event96021
    frameStart := 95977 },
  { event := event96022
    frameStart := 95977 },
  { event := event96023
    frameStart := 95977 },
  { event := event96024
    frameStart := 95977 },
  { event := event96025
    frameStart := 95977 },
  { event := event96026
    frameStart := 95977 },
  { event := event96027
    frameStart := 95977 },
  { event := event96028
    frameStart := 95977 },
  { event := event96029
    frameStart := 95977 },
  { event := event96030
    frameStart := 95977 },
  { event := event96031
    frameStart := 95977 }
]

def eventLeaf6002 : Array AnnotatedEvent := #[
  { event := event96032
    frameStart := 95977 },
  { event := event96033
    frameStart := 95977 },
  { event := event96034
    frameStart := 95977 },
  { event := event96035
    frameStart := 95977 },
  { event := event96036
    frameStart := 95977 },
  { event := event96037
    frameStart := 95977 },
  { event := event96038
    frameStart := 95977 },
  { event := event96039
    frameStart := 95977 },
  { event := event96040
    frameStart := 95977 },
  { event := event96041
    frameStart := 95977 },
  { event := event96042
    frameStart := 95977 },
  { event := event96043
    frameStart := 95977 },
  { event := event96044
    frameStart := 95977 },
  { event := event96045
    frameStart := 95977 },
  { event := event96046
    frameStart := 95977 },
  { event := event96047
    frameStart := 95977 }
]

def eventLeaf6003 : Array AnnotatedEvent := #[
  { event := event96048
    frameStart := 95977 },
  { event := event96049
    frameStart := 95977 },
  { event := event96050
    frameStart := 95977 },
  { event := event96051
    frameStart := 95977 },
  { event := event96052
    frameStart := 95977 },
  { event := event96053
    frameStart := 95977 },
  { event := event96054
    frameStart := 95977 },
  { event := event96055
    frameStart := 95977 },
  { event := event96056
    frameStart := 95977 },
  { event := event96057
    frameStart := 95977 },
  { event := event96058
    frameStart := 95977 },
  { event := event96059
    frameStart := 95977 },
  { event := event96060
    frameStart := 95977 },
  { event := event96061
    frameStart := 95977 },
  { event := event96062
    frameStart := 95977 },
  { event := event96063
    frameStart := 95977 }
]

def eventLeaf6004 : Array AnnotatedEvent := #[
  { event := event96064
    frameStart := 95977 },
  { event := event96065
    frameStart := 95977 },
  { event := event96066
    frameStart := 95977 },
  { event := event96067
    frameStart := 95977 },
  { event := event96068
    frameStart := 95977 },
  { event := event96069
    frameStart := 95977 },
  { event := event96070
    frameStart := 95977 },
  { event := event96071
    frameStart := 95977 },
  { event := event96072
    frameStart := 95977 },
  { event := event96073
    frameStart := 95977 },
  { event := event96074
    frameStart := 95977 },
  { event := event96075
    frameStart := 95977 },
  { event := event96076
    frameStart := 95977 },
  { event := event96077
    frameStart := 95977 },
  { event := event96078
    frameStart := 95977 },
  { event := event96079
    frameStart := 95977 }
]

def eventLeaf6005 : Array AnnotatedEvent := #[
  { event := event96080
    frameStart := 95977 },
  { event := event96081
    frameStart := 95977 },
  { event := event96082
    frameStart := 95977 },
  { event := event96083
    frameStart := 95977 },
  { event := event96084
    frameStart := 95977 },
  { event := event96085
    frameStart := 95977 },
  { event := event96086
    frameStart := 95977 },
  { event := event96087
    frameStart := 95977 },
  { event := event96088
    frameStart := 95977 },
  { event := event96089
    frameStart := 95977 },
  { event := event96090
    frameStart := 95977 },
  { event := event96091
    frameStart := 95977 },
  { event := event96092
    frameStart := 95977 },
  { event := event96093
    frameStart := 95977 },
  { event := event96094
    frameStart := 95977 },
  { event := event96095
    frameStart := 0 }
]

def eventLeaf6006 : Array AnnotatedEvent := #[
  { event := event96096
    frameStart := 0 },
  { event := event96097
    frameStart := 0 },
  { event := event96098
    frameStart := 0 },
  { event := event96099
    frameStart := 0 },
  { event := event96100
    frameStart := 0 },
  { event := event96101
    frameStart := 0 },
  { event := event96102
    frameStart := 0 },
  { event := event96103
    frameStart := 0 },
  { event := event96104
    frameStart := 0 },
  { event := event96105
    frameStart := 0 },
  { event := event96106
    frameStart := 0 },
  { event := event96107
    frameStart := 0 },
  { event := event96108
    frameStart := 0 },
  { event := event96109
    frameStart := 0 },
  { event := event96110
    frameStart := 0 },
  { event := event96111
    frameStart := 0 }
]

def eventLeaf6007 : Array AnnotatedEvent := #[
  { event := event96112
    frameStart := 0 },
  { event := event96113
    frameStart := 0 },
  { event := event96114
    frameStart := 0 },
  { event := event96115
    frameStart := 0 },
  { event := event96116
    frameStart := 0 },
  { event := event96117
    frameStart := 0 },
  { event := event96118
    frameStart := 0 },
  { event := event96119
    frameStart := 0 },
  { event := event96120
    frameStart := 0 },
  { event := event96121
    frameStart := 0 },
  { event := event96122
    frameStart := 0 },
  { event := event96123
    frameStart := 0 },
  { event := event96124
    frameStart := 0 },
  { event := event96125
    frameStart := 0 },
  { event := event96126
    frameStart := 0 },
  { event := event96127
    frameStart := 0 }
]

def eventLeaf6008 : Array AnnotatedEvent := #[
  { event := event96128
    frameStart := 0 },
  { event := event96129
    frameStart := 0 },
  { event := event96130
    frameStart := 0 },
  { event := event96131
    frameStart := 0 },
  { event := event96132
    frameStart := 96132 },
  { event := event96133
    frameStart := 96132 },
  { event := event96134
    frameStart := 96132 },
  { event := event96135
    frameStart := 96132 },
  { event := event96136
    frameStart := 96132 },
  { event := event96137
    frameStart := 96132 },
  { event := event96138
    frameStart := 96132 },
  { event := event96139
    frameStart := 96132 },
  { event := event96140
    frameStart := 96132 },
  { event := event96141
    frameStart := 96132 },
  { event := event96142
    frameStart := 96132 },
  { event := event96143
    frameStart := 96132 }
]

def eventLeaf6009 : Array AnnotatedEvent := #[
  { event := event96144
    frameStart := 96132 },
  { event := event96145
    frameStart := 96132 },
  { event := event96146
    frameStart := 96132 },
  { event := event96147
    frameStart := 96132 },
  { event := event96148
    frameStart := 96132 },
  { event := event96149
    frameStart := 96132 },
  { event := event96150
    frameStart := 96132 },
  { event := event96151
    frameStart := 96132 },
  { event := event96152
    frameStart := 96132 },
  { event := event96153
    frameStart := 96132 },
  { event := event96154
    frameStart := 96132 },
  { event := event96155
    frameStart := 96132 },
  { event := event96156
    frameStart := 96132 },
  { event := event96157
    frameStart := 96132 },
  { event := event96158
    frameStart := 96132 },
  { event := event96159
    frameStart := 96132 }
]

def eventLeaf6010 : Array AnnotatedEvent := #[
  { event := event96160
    frameStart := 96132 },
  { event := event96161
    frameStart := 96132 },
  { event := event96162
    frameStart := 96132 },
  { event := event96163
    frameStart := 96132 },
  { event := event96164
    frameStart := 96132 },
  { event := event96165
    frameStart := 96132 },
  { event := event96166
    frameStart := 96132 },
  { event := event96167
    frameStart := 96132 },
  { event := event96168
    frameStart := 96132 },
  { event := event96169
    frameStart := 96132 },
  { event := event96170
    frameStart := 96132 },
  { event := event96171
    frameStart := 96132 },
  { event := event96172
    frameStart := 96132 },
  { event := event96173
    frameStart := 96132 },
  { event := event96174
    frameStart := 96132 },
  { event := event96175
    frameStart := 96132 }
]

def eventLeaf6011 : Array AnnotatedEvent := #[
  { event := event96176
    frameStart := 96132 },
  { event := event96177
    frameStart := 96132 },
  { event := event96178
    frameStart := 96132 },
  { event := event96179
    frameStart := 96132 },
  { event := event96180
    frameStart := 96132 },
  { event := event96181
    frameStart := 96132 },
  { event := event96182
    frameStart := 96132 },
  { event := event96183
    frameStart := 96132 },
  { event := event96184
    frameStart := 96132 },
  { event := event96185
    frameStart := 96132 },
  { event := event96186
    frameStart := 96186 },
  { event := event96187
    frameStart := 96186 },
  { event := event96188
    frameStart := 96186 },
  { event := event96189
    frameStart := 96186 },
  { event := event96190
    frameStart := 96186 },
  { event := event96191
    frameStart := 96186 }
]

def eventLeaf6012 : Array AnnotatedEvent := #[
  { event := event96192
    frameStart := 96186 },
  { event := event96193
    frameStart := 96186 },
  { event := event96194
    frameStart := 96186 },
  { event := event96195
    frameStart := 96186 },
  { event := event96196
    frameStart := 96186 },
  { event := event96197
    frameStart := 96186 },
  { event := event96198
    frameStart := 96186 },
  { event := event96199
    frameStart := 96186 },
  { event := event96200
    frameStart := 96186 },
  { event := event96201
    frameStart := 96186 },
  { event := event96202
    frameStart := 96186 },
  { event := event96203
    frameStart := 96186 },
  { event := event96204
    frameStart := 96186 },
  { event := event96205
    frameStart := 96186 },
  { event := event96206
    frameStart := 96186 },
  { event := event96207
    frameStart := 96186 }
]

def eventLeaf6013 : Array AnnotatedEvent := #[
  { event := event96208
    frameStart := 96186 },
  { event := event96209
    frameStart := 96186 },
  { event := event96210
    frameStart := 96186 },
  { event := event96211
    frameStart := 96186 },
  { event := event96212
    frameStart := 96186 },
  { event := event96213
    frameStart := 96186 },
  { event := event96214
    frameStart := 96186 },
  { event := event96215
    frameStart := 96186 },
  { event := event96216
    frameStart := 96186 },
  { event := event96217
    frameStart := 96186 },
  { event := event96218
    frameStart := 96186 },
  { event := event96219
    frameStart := 96186 },
  { event := event96220
    frameStart := 96186 },
  { event := event96221
    frameStart := 96186 },
  { event := event96222
    frameStart := 96186 },
  { event := event96223
    frameStart := 96186 }
]

def eventLeaf6014 : Array AnnotatedEvent := #[
  { event := event96224
    frameStart := 96186 },
  { event := event96225
    frameStart := 96186 },
  { event := event96226
    frameStart := 96186 },
  { event := event96227
    frameStart := 96186 },
  { event := event96228
    frameStart := 96186 },
  { event := event96229
    frameStart := 96186 },
  { event := event96230
    frameStart := 96186 },
  { event := event96231
    frameStart := 96186 },
  { event := event96232
    frameStart := 96186 },
  { event := event96233
    frameStart := 96186 },
  { event := event96234
    frameStart := 96186 },
  { event := event96235
    frameStart := 96186 },
  { event := event96236
    frameStart := 96186 },
  { event := event96237
    frameStart := 96186 },
  { event := event96238
    frameStart := 96186 },
  { event := event96239
    frameStart := 96186 }
]

def eventLeaf6015 : Array AnnotatedEvent := #[
  { event := event96240
    frameStart := 96186 },
  { event := event96241
    frameStart := 96186 },
  { event := event96242
    frameStart := 96186 },
  { event := event96243
    frameStart := 96186 },
  { event := event96244
    frameStart := 96186 },
  { event := event96245
    frameStart := 96186 },
  { event := event96246
    frameStart := 96186 },
  { event := event96247
    frameStart := 96186 },
  { event := event96248
    frameStart := 96186 },
  { event := event96249
    frameStart := 96186 },
  { event := event96250
    frameStart := 96186 },
  { event := event96251
    frameStart := 96186 },
  { event := event96252
    frameStart := 96186 },
  { event := event96253
    frameStart := 96186 },
  { event := event96254
    frameStart := 96186 },
  { event := event96255
    frameStart := 96186 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events375
