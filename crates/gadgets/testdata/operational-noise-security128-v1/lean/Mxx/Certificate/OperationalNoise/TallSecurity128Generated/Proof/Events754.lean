import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events754

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event193024 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47882⟩⟩) (.authority (.programFamilyFact))

def exact193025RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47882⟩⟩], []⟩, (1)⟩]

theorem exact193025RawTermsValid :
    exact193025RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193025 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47882⟩⟩) exact193025RawTerms (.finite 60) 193024 .exactZero (none)

def event193026 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15111⟩⟩) 0 ⟨5905⟩ 193022

def event193027 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15111⟩⟩) (.authority (.programFamilyFact))

def exact193028RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩], []⟩, (1)⟩]

theorem exact193028RawTermsValid :
    exact193028RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193028 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15111⟩⟩) exact193028RawTerms (.finite 60) 193027 .exactZero (none)

def event193029 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47883⟩⟩) 0 ⟨15111⟩ 193028

def event193030 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47883⟩⟩) 1 ⟨47882⟩ 193025

def event193031 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47883⟩⟩) (.product (.predecessor 0 193029 .coefficient) (.predecessor 1 193030 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event193032 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47883⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], []⟩) [⟨.result 193028 .coefficient, true, some 1⟩, ⟨.result 193025 .coefficient, true, some 1⟩])

def event193033 : Event := .survivorFold (1) 193032

def exact193034RawTerms : List Term := []

theorem exact193034RawTermsValid :
    exact193034RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193034 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47883⟩⟩) exact193034RawTerms (.finite 3600) 193031 (.finite 3600) (some (193032))

def event193035 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47884⟩⟩) 0 ⟨47883⟩ 193034

def event193036 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47884⟩⟩) (.identity (.predecessor 0 193035 .coefficient))

def event193037 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47884⟩⟩) (.finite 3600)

def event193038 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48609⟩⟩) 0 ⟨47884⟩ 193037

def event193039 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48609⟩⟩) (.authority (.relationPreimageSource ⟨54⟩))

def exact193040RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48609⟩⟩]⟩, (1)⟩]

theorem exact193040RawTermsValid :
    exact193040RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193040 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48609⟩⟩) exact193040RawTerms (.finite 5647228698) 193039 .exactZero (none)

def event193041 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact193042RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact193042RawTermsValid :
    exact193042RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193042 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact193042RawTerms .large 193041 .exactZero (none)

def event193043 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48610⟩⟩) 0 ⟨35⟩ 193042

def event193044 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48610⟩⟩) 1 ⟨48609⟩ 193040

def event193045 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48610⟩⟩) (.product (.predecessor 0 193043 .coefficient) (.predecessor 1 193044 .coefficient) (⟨false, false, none, none, none⟩))

def event193046 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48610⟩⟩, .operator (⟨193042, 0⟩, ⟨193040, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48609⟩⟩]⟩, (1)⟩)

def exact193047RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48609⟩⟩]⟩, (1)⟩]

theorem exact193047RawTermsValid :
    exact193047RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193047 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48610⟩⟩) exact193047RawTerms .large 193045 .exactZero (none)

def event193048 : Event := .preFoldPolynomial 193047 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48609⟩⟩]⟩, (1)⟩] .exactZero none

def exact193049RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48609⟩⟩]⟩, (1)⟩]

def event193049 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨48610⟩⟩) 193048 exact193049RawTerms .large 193045 .exactZero (none)

def event193050 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨49685⟩⟩)

def event193051 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event193052 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event193053 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5893⟩⟩) (.authority (.operator))

def event193054 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5893⟩⟩) (.finite 7)

def event193055 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event193056 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event193057 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event193058 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event193059 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 193058

def event193060 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 193056

def event193061 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 193059 .coefficient) (.value (.predecessor 1 193060 .coefficient)))

def event193062 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event193063 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5895⟩⟩) 0 ⟨392⟩ 193062

def event193064 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5895⟩⟩) 1 ⟨5893⟩ 193054

def event193065 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5895⟩⟩) (.sum [.predecessor 0 193063 .coefficient, .predecessor 1 193064 .coefficient])

def event193066 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5895⟩⟩) (.finite 655347)

def event193067 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5905⟩⟩) 0 ⟨5895⟩ 193066

def event193068 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5905⟩⟩) 1 ⟨5426⟩ 193052

def event193069 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5905⟩⟩) (.identity (.predecessor 1 193068 .coefficient))

def event193070 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5905⟩⟩) (.finite 655360)

def event193071 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47882⟩⟩) 0 ⟨5905⟩ 193070

def event193072 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47882⟩⟩) (.authority (.programFamilyFact))

def exact193073RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47882⟩⟩], []⟩, (1)⟩]

theorem exact193073RawTermsValid :
    exact193073RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193073 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47882⟩⟩) exact193073RawTerms (.finite 60) 193072 .exactZero (none)

def event193074 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15111⟩⟩) 0 ⟨5905⟩ 193070

def event193075 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15111⟩⟩) (.authority (.programFamilyFact))

def exact193076RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩], []⟩, (1)⟩]

theorem exact193076RawTermsValid :
    exact193076RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193076 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15111⟩⟩) exact193076RawTerms (.finite 60) 193075 .exactZero (none)

def event193077 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47883⟩⟩) 0 ⟨15111⟩ 193076

def event193078 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47883⟩⟩) 1 ⟨47882⟩ 193073

def event193079 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47883⟩⟩) (.product (.predecessor 0 193077 .coefficient) (.predecessor 1 193078 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event193080 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨47883⟩⟩, .operator (⟨193076, 0⟩, ⟨193073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], []⟩, (1)⟩)

def exact193081RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], []⟩, (1)⟩]

theorem exact193081RawTermsValid :
    exact193081RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193081 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47883⟩⟩) exact193081RawTerms (.finite 3600) 193079 .exactZero (none)

def event193082 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47884⟩⟩) 0 ⟨47883⟩ 193081

def event193083 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47884⟩⟩) (.identity (.predecessor 0 193082 .coefficient))

def event193084 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47884⟩⟩) (.finite 3600)

def event193085 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49160⟩⟩) 0 ⟨47884⟩ 193084

def event193086 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49160⟩⟩) (.authority (.programFamilyFact))

def event193087 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49160⟩⟩) (.finite 3720)

def event193088 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event193089 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49161⟩⟩) 0 ⟨7177⟩ 193088

def event193090 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49161⟩⟩) 1 ⟨49160⟩ 193087

def event193091 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49161⟩⟩) (.authority (.operator))

def exact193092RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49161⟩⟩]⟩, (1)⟩]

theorem exact193092RawTermsValid :
    exact193092RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193092 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49161⟩⟩) exact193092RawTerms .large 193091 .exactZero (none)

def event193093 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49681⟩⟩) 0 ⟨49161⟩ 193092

def event193094 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49681⟩⟩) (.authority (.operator))

def exact193095RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩, (1)⟩]

theorem exact193095RawTermsValid :
    exact193095RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193095 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49681⟩⟩) exact193095RawTerms (.finite 8192) 193094 .exactZero (none)

def event193096 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event193097 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event193098 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49434⟩⟩) 0 ⟨47884⟩ 193084

def event193099 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49434⟩⟩) 1 ⟨136⟩ 193097

def event193100 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49434⟩⟩) (.sum [.predecessor 0 193098 .coefficient, .predecessor 1 193099 .coefficient])

def event193101 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨49434⟩⟩) (.finite 3600)

def event193102 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49435⟩⟩) 0 ⟨49434⟩ 193101

def event193103 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49435⟩⟩) (.identity (.predecessor 0 193102 .coefficient))

def exact193104RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], []⟩, (1)⟩]

theorem exact193104RawTermsValid :
    exact193104RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193104 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49435⟩⟩) exact193104RawTerms (.finite 3600) 193103 .exactZero (none)

def event193105 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact193106RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact193106RawTermsValid :
    exact193106RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193106 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact193106RawTerms .large 193105 .exactZero (none)

def event193107 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49436⟩⟩) 0 ⟨6908⟩ 193106

def event193108 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49436⟩⟩) 1 ⟨49435⟩ 193104

def event193109 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49436⟩⟩) (.product (.predecessor 0 193107 .coefficient) (.predecessor 1 193108 .coefficient) (⟨false, false, none, none, none⟩))

def event193110 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49436⟩⟩, .operator (⟨193106, 0⟩, ⟨193104, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact193111RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact193111RawTermsValid :
    exact193111RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193111 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49436⟩⟩) exact193111RawTerms .large 193109 .exactZero (none)

def event193112 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.authority (.operator))

def event193113 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2370⟩⟩) (.finite 1)

def event193114 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7178⟩⟩) 0 ⟨7177⟩ 193088

def event193115 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7178⟩⟩) (.authority (.operator))

def exact193116RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7178⟩⟩]⟩, (1)⟩]

theorem exact193116RawTermsValid :
    exact193116RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193116 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7178⟩⟩) exact193116RawTerms .large 193115 .exactZero (none)

def event193117 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7285⟩⟩) 0 ⟨7178⟩ 193116

def event193118 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7285⟩⟩) (.identity (.predecessor 0 193117 .coefficient))

def exact193119RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩, (1)⟩]

theorem exact193119RawTermsValid :
    exact193119RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193119 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7285⟩⟩) exact193119RawTerms .large 193118 .exactZero (none)

def event193120 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9565⟩⟩) 0 ⟨7285⟩ 193119

def event193121 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9565⟩⟩) (.authority (.operator))

def exact193122RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩]

theorem exact193122RawTermsValid :
    exact193122RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193122 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9565⟩⟩) exact193122RawTerms (.finite 8192) 193121 .exactZero (none)

def event193123 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9566⟩⟩) 0 ⟨9565⟩ 193122

def event193124 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9566⟩⟩) 1 ⟨2370⟩ 193113

def event193125 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9566⟩⟩) (.scale (.predecessor 0 193123 .coefficient) (.value (.predecessor 1 193124 .coefficient)))

def exact193126RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩]

theorem exact193126RawTermsValid :
    exact193126RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193126 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9566⟩⟩) exact193126RawTerms (.finite 8192) 193125 .exactZero (none)

def event193127 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7302⟩⟩) 0 ⟨7178⟩ 193116

def event193128 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7302⟩⟩) (.identity (.predecessor 0 193127 .coefficient))

def exact193129RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩, (1)⟩]

theorem exact193129RawTermsValid :
    exact193129RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193129 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7302⟩⟩) exact193129RawTerms .large 193128 .exactZero (none)

def event193130 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9567⟩⟩) 0 ⟨7302⟩ 193129

def event193131 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9567⟩⟩) 1 ⟨9566⟩ 193126

def event193132 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9567⟩⟩) (.product (.predecessor 0 193130 .coefficient) (.predecessor 1 193131 .coefficient) (⟨false, false, none, none, none⟩))

def event193133 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨9567⟩⟩, .operator (⟨193129, 0⟩, ⟨193126, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩)

def exact193134RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩]

theorem exact193134RawTermsValid :
    exact193134RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193134 : Event := .resultExact (⟨.program ⟨257⟩, ⟨9567⟩⟩) exact193134RawTerms .large 193132 .exactZero (none)

def event193135 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49437⟩⟩) 0 ⟨9567⟩ 193134

def event193136 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49437⟩⟩) 1 ⟨49436⟩ 193111

def event193137 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49437⟩⟩) (.sum [.predecessor 0 193135 .coefficient, .predecessor 1 193136 .coefficient])

def exact193138RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact193138RawTermsValid :
    exact193138RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193138 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49437⟩⟩) exact193138RawTerms .large 193137 .exactZero (none)

def event193139 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49684⟩⟩) 0 ⟨49437⟩ 193138

def event193140 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49684⟩⟩) 1 ⟨49681⟩ 193095

def event193141 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49684⟩⟩) (.product (.predecessor 0 193139 .coefficient) (.predecessor 1 193140 .coefficient) (⟨false, false, none, none, none⟩))

def event193142 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49684⟩⟩, .operator (⟨193138, 0⟩, ⟨193095, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩, (1)⟩)

def event193143 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49684⟩⟩, .operator (⟨193138, 1⟩, ⟨193095, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩, (-1)⟩)

def event193144 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨49684⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49681⟩⟩) ⟨49161⟩ 193092)

def event193145 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49684⟩⟩, .relation 193144 0, ⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨49161⟩⟩]⟩, (-1)⟩)

def exact193146RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨49161⟩⟩]⟩, (-1)⟩]

theorem exact193146RawTermsValid :
    exact193146RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193146 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49684⟩⟩) exact193146RawTerms .large 193141 .exactZero (none)

def event193147 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48164⟩⟩) 0 ⟨47884⟩ 193084

def event193148 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48164⟩⟩) (.authority (.programFamilyFact))

def exact193149RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48164⟩⟩], []⟩, (1)⟩]

theorem exact193149RawTermsValid :
    exact193149RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193149 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48164⟩⟩) exact193149RawTerms (.finite 60) 193148 .exactZero (none)

def event193150 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48166⟩⟩) 0 ⟨6908⟩ 193106

def event193151 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48166⟩⟩) 1 ⟨48164⟩ 193149

def event193152 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48166⟩⟩) (.product (.predecessor 0 193150 .coefficient) (.predecessor 1 193151 .coefficient) (⟨false, true, none, none, some 1⟩))

def event193153 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48166⟩⟩, .operator (⟨193106, 0⟩, ⟨193149, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact193154RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact193154RawTermsValid :
    exact193154RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193154 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48166⟩⟩) exact193154RawTerms .large 193152 .exactZero (none)

def event193155 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7196⟩⟩) 0 ⟨7177⟩ 193088

def event193156 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7196⟩⟩) (.authority (.operator))

def exact193157RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩]

theorem exact193157RawTermsValid :
    exact193157RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193157 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7196⟩⟩) exact193157RawTerms .large 193156 .exactZero (none)

def event193158 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48167⟩⟩) 0 ⟨7196⟩ 193157

def event193159 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48167⟩⟩) 1 ⟨48166⟩ 193154

def event193160 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48167⟩⟩) (.sum [.predecessor 0 193158 .coefficient, .predecessor 1 193159 .coefficient])

def exact193161RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact193161RawTermsValid :
    exact193161RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193161 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48167⟩⟩) exact193161RawTerms .large 193160 .exactZero (none)

def event193162 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49685⟩⟩) 0 ⟨48167⟩ 193161

def event193163 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49685⟩⟩) 1 ⟨49684⟩ 193146

def event193164 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49685⟩⟩) (.sum [.predecessor 0 193162 .coefficient, .predecessor 1 193163 .coefficient])

def exact193165RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨49161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact193165RawTermsValid :
    exact193165RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193165 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49685⟩⟩) exact193165RawTerms .large 193164 .exactZero (none)

def event193166 : Event := .preFoldPolynomial 193165 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨49161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact193167RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨49161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event193167 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨49685⟩⟩) 193166 exact193167RawTerms .large 193164 .exactZero (none)

def event193168 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨47884⟩⟩) ⟨⟨75⟩, ⟨54⟩, ⟨135⟩⟩ ⟨193002, 193168⟩

def event193169 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨48612⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48609⟩⟩]⟩) (1) 0 2 (.universal 193168 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48609⟩⟩]⟩) (none) 193167)

def event193170 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48612⟩⟩, .relation 193169 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩)

def event193171 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48612⟩⟩, .relation 193169 1, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩, (-1)⟩)

def event193172 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48612⟩⟩, .relation 193169 2, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨49161⟩⟩]⟩, (1)⟩)

def event193173 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48612⟩⟩, .relation 193169 3, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact193174RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨49161⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact193174RawTermsValid :
    exact193174RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193174 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48612⟩⟩) exact193174RawTerms .large 192998 (.finite 202072841853861888) (some (193000))

def event193175 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49683⟩⟩) 0 ⟨48612⟩ 193174

def event193176 : Event := .predecessor (⟨.program ⟨257⟩, ⟨49683⟩⟩) 1 ⟨49682⟩ 192977

def event193177 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49683⟩⟩) (.sum [.predecessor 0 193175 .coefficient, .predecessor 1 193176 .coefficient])

def event193178 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49683⟩⟩, .operator (⟨193174, 2⟩, ⟨192977, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨49161⟩⟩]⟩, (-1)⟩)

def event193179 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨49683⟩⟩, .operator (⟨193174, 1⟩, ⟨192977, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩, (1)⟩)

def event193180 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨49683⟩⟩) (.sum [.result 193174 .summary, .result 192977 .summary])

def exact193181RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact193181RawTermsValid :
    exact193181RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193181 : Event := .resultExact (⟨.program ⟨257⟩, ⟨49683⟩⟩) exact193181RawTerms .large 193177 (.finite 2998346861024241778688) (some (193180))

def event193182 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50081⟩⟩) 0 ⟨49683⟩ 193181

def event193183 : Event := .predecessor (⟨.program ⟨257⟩, ⟨50081⟩⟩) 1 ⟨50079⟩ 192888

def event193184 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50081⟩⟩) (.product (.predecessor 0 193182 .coefficient) (.predecessor 1 193183 .coefficient) (⟨false, false, none, none, none⟩))

def event193185 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50081⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨50079⟩⟩]⟩) [⟨.result 192888 .coefficient, false, none⟩])

def event193186 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨50081⟩⟩) (.product (.result 193181 .summary) (.transfer 193185) (⟨false, false, none, none, none⟩))

def event193187 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50081⟩⟩, .operator (⟨193181, 0⟩, ⟨192888, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50079⟩⟩]⟩, (1)⟩)

def event193188 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50081⟩⟩, .operator (⟨193181, 1⟩, ⟨192888, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50079⟩⟩]⟩, (-1)⟩)

def event193189 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨50081⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50079⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨50079⟩⟩) ⟨49319⟩ 192885)

def event193190 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨50081⟩⟩, .relation 193189 0, ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨49319⟩⟩]⟩, (-1)⟩)

def exact193191RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50079⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨49319⟩⟩]⟩, (-1)⟩]

theorem exact193191RawTermsValid :
    exact193191RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193191 : Event := .resultExact (⟨.program ⟨257⟩, ⟨50081⟩⟩) exact193191RawTerms .large 193184 (.finite 32194504275408438756654574469120) (some (193186))

def event193192 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48936⟩⟩) 0 ⟨48165⟩ 9087

def event193193 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48936⟩⟩) (.authority (.relationPreimageSource ⟨94⟩))

def exact193194RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48936⟩⟩]⟩, (1)⟩]

theorem exact193194RawTermsValid :
    exact193194RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193194 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48936⟩⟩) exact193194RawTerms (.finite 5647228698) 193193 .exactZero (none)

def event193195 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48938⟩⟩) 0 ⟨48936⟩ 193194

def event193196 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48938⟩⟩) 1 ⟨2370⟩ 4

def event193197 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48938⟩⟩) (.scale (.predecessor 0 193195 .coefficient) (.value (.predecessor 1 193196 .coefficient)))

def exact193198RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48936⟩⟩]⟩, (1)⟩]

theorem exact193198RawTermsValid :
    exact193198RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193198 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48938⟩⟩) exact193198RawTerms (.finite 5647228698) 193197 .exactZero (none)

def event193199 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48939⟩⟩) 0 ⟨5909⟩ 192995

def event193200 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48939⟩⟩) 1 ⟨48938⟩ 193198

def event193201 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48939⟩⟩) (.product (.predecessor 0 193199 .coefficient) (.predecessor 1 193200 .coefficient) (⟨false, false, none, none, none⟩))

def event193202 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48939⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨48936⟩⟩]⟩) [⟨.result 193194 .coefficient, false, none⟩])

def event193203 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48939⟩⟩) (.product (.result 192995 .summary) (.transfer 193202) (⟨false, false, none, none, none⟩))

def event193204 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48939⟩⟩, .operator (⟨192995, 0⟩, ⟨193198, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48936⟩⟩]⟩, (1)⟩)

def event193205 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨48937⟩⟩)

def event193206 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event193207 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event193208 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5893⟩⟩) (.authority (.operator))

def event193209 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5893⟩⟩) (.finite 7)

def event193210 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event193211 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event193212 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event193213 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event193214 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 193213

def event193215 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 193211

def event193216 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 193214 .coefficient) (.value (.predecessor 1 193215 .coefficient)))

def event193217 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event193218 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5895⟩⟩) 0 ⟨392⟩ 193217

def event193219 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5895⟩⟩) 1 ⟨5893⟩ 193209

def event193220 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5895⟩⟩) (.sum [.predecessor 0 193218 .coefficient, .predecessor 1 193219 .coefficient])

def event193221 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5895⟩⟩) (.finite 655347)

def event193222 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5905⟩⟩) 0 ⟨5895⟩ 193221

def event193223 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5905⟩⟩) 1 ⟨5426⟩ 193207

def event193224 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5905⟩⟩) (.identity (.predecessor 1 193223 .coefficient))

def event193225 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5905⟩⟩) (.finite 655360)

def event193226 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47882⟩⟩) 0 ⟨5905⟩ 193225

def event193227 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47882⟩⟩) (.authority (.programFamilyFact))

def exact193228RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47882⟩⟩], []⟩, (1)⟩]

theorem exact193228RawTermsValid :
    exact193228RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193228 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47882⟩⟩) exact193228RawTerms (.finite 60) 193227 .exactZero (none)

def event193229 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15111⟩⟩) 0 ⟨5905⟩ 193225

def event193230 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15111⟩⟩) (.authority (.programFamilyFact))

def exact193231RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩], []⟩, (1)⟩]

theorem exact193231RawTermsValid :
    exact193231RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193231 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15111⟩⟩) exact193231RawTerms (.finite 60) 193230 .exactZero (none)

def event193232 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47883⟩⟩) 0 ⟨15111⟩ 193231

def event193233 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47883⟩⟩) 1 ⟨47882⟩ 193228

def event193234 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47883⟩⟩) (.product (.predecessor 0 193232 .coefficient) (.predecessor 1 193233 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event193235 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47883⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], []⟩) [⟨.result 193231 .coefficient, true, some 1⟩, ⟨.result 193228 .coefficient, true, some 1⟩])

def event193236 : Event := .survivorFold (1) 193235

def exact193237RawTerms : List Term := []

theorem exact193237RawTermsValid :
    exact193237RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193237 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47883⟩⟩) exact193237RawTerms (.finite 3600) 193234 (.finite 3600) (some (193235))

def event193238 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47884⟩⟩) 0 ⟨47883⟩ 193237

def event193239 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47884⟩⟩) (.identity (.predecessor 0 193238 .coefficient))

def event193240 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨47884⟩⟩) (.finite 3600)

def event193241 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48164⟩⟩) 0 ⟨47884⟩ 193240

def event193242 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48164⟩⟩) (.authority (.programFamilyFact))

def exact193243RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨48164⟩⟩], []⟩, (1)⟩]

theorem exact193243RawTermsValid :
    exact193243RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193243 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48164⟩⟩) exact193243RawTerms (.finite 60) 193242 .exactZero (none)

def event193244 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48165⟩⟩) 0 ⟨48164⟩ 193243

def event193245 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48165⟩⟩) (.identity (.predecessor 0 193244 .coefficient))

def event193246 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨48165⟩⟩) (.finite 60)

def event193247 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48936⟩⟩) 0 ⟨48165⟩ 193246

def event193248 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48936⟩⟩) (.authority (.relationPreimageSource ⟨94⟩))

def exact193249RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨48936⟩⟩]⟩, (1)⟩]

theorem exact193249RawTermsValid :
    exact193249RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193249 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48936⟩⟩) exact193249RawTerms (.finite 5647228698) 193248 .exactZero (none)

def event193250 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact193251RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact193251RawTermsValid :
    exact193251RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193251 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact193251RawTerms .large 193250 .exactZero (none)

def event193252 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48937⟩⟩) 0 ⟨35⟩ 193251

def event193253 : Event := .predecessor (⟨.program ⟨257⟩, ⟨48937⟩⟩) 1 ⟨48936⟩ 193249

def event193254 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨48937⟩⟩) (.product (.predecessor 0 193252 .coefficient) (.predecessor 1 193253 .coefficient) (⟨false, false, none, none, none⟩))

def event193255 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨48937⟩⟩, .operator (⟨193251, 0⟩, ⟨193249, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48936⟩⟩]⟩, (1)⟩)

def exact193256RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48936⟩⟩]⟩, (1)⟩]

theorem exact193256RawTermsValid :
    exact193256RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event193256 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48937⟩⟩) exact193256RawTerms .large 193254 .exactZero (none)

def event193257 : Event := .preFoldPolynomial 193256 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48936⟩⟩]⟩, (1)⟩] .exactZero none

def exact193258RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48936⟩⟩]⟩, (1)⟩]

def event193258 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨48937⟩⟩) 193257 exact193258RawTerms .large 193254 .exactZero (none)

def event193259 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨50083⟩⟩)

def event193260 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event193261 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event193262 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5893⟩⟩) (.authority (.operator))

def event193263 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5893⟩⟩) (.finite 7)

def event193264 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event193265 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event193266 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event193267 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event193268 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 193267

def event193269 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 193265

def event193270 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 193268 .coefficient) (.value (.predecessor 1 193269 .coefficient)))

def event193271 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event193272 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5895⟩⟩) 0 ⟨392⟩ 193271

def event193273 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5895⟩⟩) 1 ⟨5893⟩ 193263

def event193274 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5895⟩⟩) (.sum [.predecessor 0 193272 .coefficient, .predecessor 1 193273 .coefficient])

def event193275 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5895⟩⟩) (.finite 655347)

def event193276 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5905⟩⟩) 0 ⟨5895⟩ 193275

def event193277 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5905⟩⟩) 1 ⟨5426⟩ 193261

def event193278 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5905⟩⟩) (.identity (.predecessor 1 193277 .coefficient))

def event193279 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5905⟩⟩) (.finite 655360)

def eventLeaf12064 : Array AnnotatedEvent := #[
  { event := event193024
    frameStart := 193002 },
  { event := event193025
    frameStart := 193002 },
  { event := event193026
    frameStart := 193002 },
  { event := event193027
    frameStart := 193002 },
  { event := event193028
    frameStart := 193002 },
  { event := event193029
    frameStart := 193002 },
  { event := event193030
    frameStart := 193002 },
  { event := event193031
    frameStart := 193002 },
  { event := event193032
    frameStart := 193002 },
  { event := event193033
    frameStart := 193002 },
  { event := event193034
    frameStart := 193002 },
  { event := event193035
    frameStart := 193002 },
  { event := event193036
    frameStart := 193002 },
  { event := event193037
    frameStart := 193002 },
  { event := event193038
    frameStart := 193002 },
  { event := event193039
    frameStart := 193002 }
]

def eventLeaf12065 : Array AnnotatedEvent := #[
  { event := event193040
    frameStart := 193002 },
  { event := event193041
    frameStart := 193002 },
  { event := event193042
    frameStart := 193002 },
  { event := event193043
    frameStart := 193002 },
  { event := event193044
    frameStart := 193002 },
  { event := event193045
    frameStart := 193002 },
  { event := event193046
    frameStart := 193002 },
  { event := event193047
    frameStart := 193002 },
  { event := event193048
    frameStart := 193002 },
  { event := event193049
    frameStart := 193002 },
  { event := event193050
    frameStart := 193050 },
  { event := event193051
    frameStart := 193050 },
  { event := event193052
    frameStart := 193050 },
  { event := event193053
    frameStart := 193050 },
  { event := event193054
    frameStart := 193050 },
  { event := event193055
    frameStart := 193050 }
]

def eventLeaf12066 : Array AnnotatedEvent := #[
  { event := event193056
    frameStart := 193050 },
  { event := event193057
    frameStart := 193050 },
  { event := event193058
    frameStart := 193050 },
  { event := event193059
    frameStart := 193050 },
  { event := event193060
    frameStart := 193050 },
  { event := event193061
    frameStart := 193050 },
  { event := event193062
    frameStart := 193050 },
  { event := event193063
    frameStart := 193050 },
  { event := event193064
    frameStart := 193050 },
  { event := event193065
    frameStart := 193050 },
  { event := event193066
    frameStart := 193050 },
  { event := event193067
    frameStart := 193050 },
  { event := event193068
    frameStart := 193050 },
  { event := event193069
    frameStart := 193050 },
  { event := event193070
    frameStart := 193050 },
  { event := event193071
    frameStart := 193050 }
]

def eventLeaf12067 : Array AnnotatedEvent := #[
  { event := event193072
    frameStart := 193050 },
  { event := event193073
    frameStart := 193050 },
  { event := event193074
    frameStart := 193050 },
  { event := event193075
    frameStart := 193050 },
  { event := event193076
    frameStart := 193050 },
  { event := event193077
    frameStart := 193050 },
  { event := event193078
    frameStart := 193050 },
  { event := event193079
    frameStart := 193050 },
  { event := event193080
    frameStart := 193050 },
  { event := event193081
    frameStart := 193050 },
  { event := event193082
    frameStart := 193050 },
  { event := event193083
    frameStart := 193050 },
  { event := event193084
    frameStart := 193050 },
  { event := event193085
    frameStart := 193050 },
  { event := event193086
    frameStart := 193050 },
  { event := event193087
    frameStart := 193050 }
]

def eventLeaf12068 : Array AnnotatedEvent := #[
  { event := event193088
    frameStart := 193050 },
  { event := event193089
    frameStart := 193050 },
  { event := event193090
    frameStart := 193050 },
  { event := event193091
    frameStart := 193050 },
  { event := event193092
    frameStart := 193050 },
  { event := event193093
    frameStart := 193050 },
  { event := event193094
    frameStart := 193050 },
  { event := event193095
    frameStart := 193050 },
  { event := event193096
    frameStart := 193050 },
  { event := event193097
    frameStart := 193050 },
  { event := event193098
    frameStart := 193050 },
  { event := event193099
    frameStart := 193050 },
  { event := event193100
    frameStart := 193050 },
  { event := event193101
    frameStart := 193050 },
  { event := event193102
    frameStart := 193050 },
  { event := event193103
    frameStart := 193050 }
]

def eventLeaf12069 : Array AnnotatedEvent := #[
  { event := event193104
    frameStart := 193050 },
  { event := event193105
    frameStart := 193050 },
  { event := event193106
    frameStart := 193050 },
  { event := event193107
    frameStart := 193050 },
  { event := event193108
    frameStart := 193050 },
  { event := event193109
    frameStart := 193050 },
  { event := event193110
    frameStart := 193050 },
  { event := event193111
    frameStart := 193050 },
  { event := event193112
    frameStart := 193050 },
  { event := event193113
    frameStart := 193050 },
  { event := event193114
    frameStart := 193050 },
  { event := event193115
    frameStart := 193050 },
  { event := event193116
    frameStart := 193050 },
  { event := event193117
    frameStart := 193050 },
  { event := event193118
    frameStart := 193050 },
  { event := event193119
    frameStart := 193050 }
]

def eventLeaf12070 : Array AnnotatedEvent := #[
  { event := event193120
    frameStart := 193050 },
  { event := event193121
    frameStart := 193050 },
  { event := event193122
    frameStart := 193050 },
  { event := event193123
    frameStart := 193050 },
  { event := event193124
    frameStart := 193050 },
  { event := event193125
    frameStart := 193050 },
  { event := event193126
    frameStart := 193050 },
  { event := event193127
    frameStart := 193050 },
  { event := event193128
    frameStart := 193050 },
  { event := event193129
    frameStart := 193050 },
  { event := event193130
    frameStart := 193050 },
  { event := event193131
    frameStart := 193050 },
  { event := event193132
    frameStart := 193050 },
  { event := event193133
    frameStart := 193050 },
  { event := event193134
    frameStart := 193050 },
  { event := event193135
    frameStart := 193050 }
]

def eventLeaf12071 : Array AnnotatedEvent := #[
  { event := event193136
    frameStart := 193050 },
  { event := event193137
    frameStart := 193050 },
  { event := event193138
    frameStart := 193050 },
  { event := event193139
    frameStart := 193050 },
  { event := event193140
    frameStart := 193050 },
  { event := event193141
    frameStart := 193050 },
  { event := event193142
    frameStart := 193050 },
  { event := event193143
    frameStart := 193050 },
  { event := event193144
    frameStart := 193050 },
  { event := event193145
    frameStart := 193050 },
  { event := event193146
    frameStart := 193050 },
  { event := event193147
    frameStart := 193050 },
  { event := event193148
    frameStart := 193050 },
  { event := event193149
    frameStart := 193050 },
  { event := event193150
    frameStart := 193050 },
  { event := event193151
    frameStart := 193050 }
]

def eventLeaf12072 : Array AnnotatedEvent := #[
  { event := event193152
    frameStart := 193050 },
  { event := event193153
    frameStart := 193050 },
  { event := event193154
    frameStart := 193050 },
  { event := event193155
    frameStart := 193050 },
  { event := event193156
    frameStart := 193050 },
  { event := event193157
    frameStart := 193050 },
  { event := event193158
    frameStart := 193050 },
  { event := event193159
    frameStart := 193050 },
  { event := event193160
    frameStart := 193050 },
  { event := event193161
    frameStart := 193050 },
  { event := event193162
    frameStart := 193050 },
  { event := event193163
    frameStart := 193050 },
  { event := event193164
    frameStart := 193050 },
  { event := event193165
    frameStart := 193050 },
  { event := event193166
    frameStart := 193050 },
  { event := event193167
    frameStart := 193050 }
]

def eventLeaf12073 : Array AnnotatedEvent := #[
  { event := event193168
    frameStart := 0 },
  { event := event193169
    frameStart := 0 },
  { event := event193170
    frameStart := 0 },
  { event := event193171
    frameStart := 0 },
  { event := event193172
    frameStart := 0 },
  { event := event193173
    frameStart := 0 },
  { event := event193174
    frameStart := 0 },
  { event := event193175
    frameStart := 0 },
  { event := event193176
    frameStart := 0 },
  { event := event193177
    frameStart := 0 },
  { event := event193178
    frameStart := 0 },
  { event := event193179
    frameStart := 0 },
  { event := event193180
    frameStart := 0 },
  { event := event193181
    frameStart := 0 },
  { event := event193182
    frameStart := 0 },
  { event := event193183
    frameStart := 0 }
]

def eventLeaf12074 : Array AnnotatedEvent := #[
  { event := event193184
    frameStart := 0 },
  { event := event193185
    frameStart := 0 },
  { event := event193186
    frameStart := 0 },
  { event := event193187
    frameStart := 0 },
  { event := event193188
    frameStart := 0 },
  { event := event193189
    frameStart := 0 },
  { event := event193190
    frameStart := 0 },
  { event := event193191
    frameStart := 0 },
  { event := event193192
    frameStart := 0 },
  { event := event193193
    frameStart := 0 },
  { event := event193194
    frameStart := 0 },
  { event := event193195
    frameStart := 0 },
  { event := event193196
    frameStart := 0 },
  { event := event193197
    frameStart := 0 },
  { event := event193198
    frameStart := 0 },
  { event := event193199
    frameStart := 0 }
]

def eventLeaf12075 : Array AnnotatedEvent := #[
  { event := event193200
    frameStart := 0 },
  { event := event193201
    frameStart := 0 },
  { event := event193202
    frameStart := 0 },
  { event := event193203
    frameStart := 0 },
  { event := event193204
    frameStart := 0 },
  { event := event193205
    frameStart := 193205 },
  { event := event193206
    frameStart := 193205 },
  { event := event193207
    frameStart := 193205 },
  { event := event193208
    frameStart := 193205 },
  { event := event193209
    frameStart := 193205 },
  { event := event193210
    frameStart := 193205 },
  { event := event193211
    frameStart := 193205 },
  { event := event193212
    frameStart := 193205 },
  { event := event193213
    frameStart := 193205 },
  { event := event193214
    frameStart := 193205 },
  { event := event193215
    frameStart := 193205 }
]

def eventLeaf12076 : Array AnnotatedEvent := #[
  { event := event193216
    frameStart := 193205 },
  { event := event193217
    frameStart := 193205 },
  { event := event193218
    frameStart := 193205 },
  { event := event193219
    frameStart := 193205 },
  { event := event193220
    frameStart := 193205 },
  { event := event193221
    frameStart := 193205 },
  { event := event193222
    frameStart := 193205 },
  { event := event193223
    frameStart := 193205 },
  { event := event193224
    frameStart := 193205 },
  { event := event193225
    frameStart := 193205 },
  { event := event193226
    frameStart := 193205 },
  { event := event193227
    frameStart := 193205 },
  { event := event193228
    frameStart := 193205 },
  { event := event193229
    frameStart := 193205 },
  { event := event193230
    frameStart := 193205 },
  { event := event193231
    frameStart := 193205 }
]

def eventLeaf12077 : Array AnnotatedEvent := #[
  { event := event193232
    frameStart := 193205 },
  { event := event193233
    frameStart := 193205 },
  { event := event193234
    frameStart := 193205 },
  { event := event193235
    frameStart := 193205 },
  { event := event193236
    frameStart := 193205 },
  { event := event193237
    frameStart := 193205 },
  { event := event193238
    frameStart := 193205 },
  { event := event193239
    frameStart := 193205 },
  { event := event193240
    frameStart := 193205 },
  { event := event193241
    frameStart := 193205 },
  { event := event193242
    frameStart := 193205 },
  { event := event193243
    frameStart := 193205 },
  { event := event193244
    frameStart := 193205 },
  { event := event193245
    frameStart := 193205 },
  { event := event193246
    frameStart := 193205 },
  { event := event193247
    frameStart := 193205 }
]

def eventLeaf12078 : Array AnnotatedEvent := #[
  { event := event193248
    frameStart := 193205 },
  { event := event193249
    frameStart := 193205 },
  { event := event193250
    frameStart := 193205 },
  { event := event193251
    frameStart := 193205 },
  { event := event193252
    frameStart := 193205 },
  { event := event193253
    frameStart := 193205 },
  { event := event193254
    frameStart := 193205 },
  { event := event193255
    frameStart := 193205 },
  { event := event193256
    frameStart := 193205 },
  { event := event193257
    frameStart := 193205 },
  { event := event193258
    frameStart := 193205 },
  { event := event193259
    frameStart := 193259 },
  { event := event193260
    frameStart := 193259 },
  { event := event193261
    frameStart := 193259 },
  { event := event193262
    frameStart := 193259 },
  { event := event193263
    frameStart := 193259 }
]

def eventLeaf12079 : Array AnnotatedEvent := #[
  { event := event193264
    frameStart := 193259 },
  { event := event193265
    frameStart := 193259 },
  { event := event193266
    frameStart := 193259 },
  { event := event193267
    frameStart := 193259 },
  { event := event193268
    frameStart := 193259 },
  { event := event193269
    frameStart := 193259 },
  { event := event193270
    frameStart := 193259 },
  { event := event193271
    frameStart := 193259 },
  { event := event193272
    frameStart := 193259 },
  { event := event193273
    frameStart := 193259 },
  { event := event193274
    frameStart := 193259 },
  { event := event193275
    frameStart := 193259 },
  { event := event193276
    frameStart := 193259 },
  { event := event193277
    frameStart := 193259 },
  { event := event193278
    frameStart := 193259 },
  { event := event193279
    frameStart := 193259 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events754
