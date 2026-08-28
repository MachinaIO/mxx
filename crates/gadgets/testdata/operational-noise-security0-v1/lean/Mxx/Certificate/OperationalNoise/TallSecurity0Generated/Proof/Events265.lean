import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events265

open Mxx.Certificate.OperationalNoise
open CertificateABI

def event67840 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19884⟩⟩) 0 ⟨12364⟩ 67839

def event67841 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19884⟩⟩) (.authority (.relationPreimageSource ⟨20⟩))

def exact67842RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨19884⟩⟩]⟩, (1)⟩]

theorem exact67842RawTermsValid :
    exact67842RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67842 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19884⟩⟩) exact67842RawTerms (.finite 136065468) 67841 .exactZero (none)

def event67843 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact67844RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact67844RawTermsValid :
    exact67844RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67844 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact67844RawTerms .large 67843 .exactZero (none)

def event67845 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19885⟩⟩) 0 ⟨6⟩ 67844

def event67846 : Event := .predecessor (⟨.program ⟨214⟩, ⟨19885⟩⟩) 1 ⟨19884⟩ 67842

def event67847 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨19885⟩⟩) (.product (.predecessor 0 67845 .coefficient) (.predecessor 1 67846 .coefficient) (⟨false, false, none, none, none⟩))

def event67848 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19885⟩⟩, .operator (⟨67844, 0⟩, ⟨67842, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19884⟩⟩]⟩, (1)⟩)

def exact67849RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19884⟩⟩]⟩, (1)⟩]

theorem exact67849RawTermsValid :
    exact67849RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67849 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19885⟩⟩) exact67849RawTerms .large 67847 .exactZero (none)

def event67850 : Event := .preFoldPolynomial 67849 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19884⟩⟩]⟩, (1)⟩] .exactZero none

def exact67851RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19884⟩⟩]⟩, (1)⟩]

def event67851 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨19885⟩⟩) 67850 exact67851RawTerms .large 67847 .exactZero (none)

def event67852 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨25372⟩⟩)

def event67853 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event67854 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event67855 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event67856 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event67857 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event67858 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event67859 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event67860 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event67861 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 67860

def event67862 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 67858

def event67863 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 67861 .coefficient) (.value (.predecessor 1 67862 .coefficient)))

def event67864 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event67865 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 67864

def event67866 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 67856

def event67867 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 67865 .coefficient, .predecessor 1 67866 .coefficient])

def event67868 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event67869 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 67868

def event67870 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 67854

def event67871 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 67870 .coefficient))

def event67872 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event67873 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12362⟩⟩) 0 ⟨5530⟩ 67872

def event67874 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12362⟩⟩) (.authority (.programFamilyFact))

def exact67875RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12362⟩⟩], []⟩, (1)⟩]

theorem exact67875RawTermsValid :
    exact67875RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67875 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12362⟩⟩) exact67875RawTerms (.finite 40) 67874 .exactZero (none)

def event67876 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9815⟩⟩) 0 ⟨5530⟩ 67872

def event67877 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9815⟩⟩) (.authority (.programFamilyFact))

def exact67878RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩], []⟩, (1)⟩]

theorem exact67878RawTermsValid :
    exact67878RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67878 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9815⟩⟩) exact67878RawTerms (.finite 40) 67877 .exactZero (none)

def event67879 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12363⟩⟩) 0 ⟨9815⟩ 67878

def event67880 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12363⟩⟩) 1 ⟨12362⟩ 67875

def event67881 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12363⟩⟩) (.product (.predecessor 0 67879 .coefficient) (.predecessor 1 67880 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event67882 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12363⟩⟩, .operator (⟨67878, 0⟩, ⟨67875, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], []⟩, (1)⟩)

def exact67883RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], []⟩, (1)⟩]

theorem exact67883RawTermsValid :
    exact67883RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67883 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12363⟩⟩) exact67883RawTerms (.finite 1600) 67881 .exactZero (none)

def event67884 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12364⟩⟩) 0 ⟨12363⟩ 67883

def event67885 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12364⟩⟩) (.identity (.predecessor 0 67884 .coefficient))

def event67886 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12364⟩⟩) (.finite 1600)

def event67887 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23203⟩⟩) 0 ⟨12364⟩ 67886

def event67888 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23203⟩⟩) (.authority (.programFamilyFact))

def event67889 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨23203⟩⟩) (.finite 3720)

def event67890 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨6689⟩⟩) .missing

def event67891 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23204⟩⟩) 0 ⟨6689⟩ 67890

def event67892 : Event := .predecessor (⟨.program ⟨214⟩, ⟨23204⟩⟩) 1 ⟨23203⟩ 67889

def event67893 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨23204⟩⟩) (.authority (.operator))

def exact67894RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨23204⟩⟩]⟩, (1)⟩]

theorem exact67894RawTermsValid :
    exact67894RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67894 : Event := .resultExact (⟨.program ⟨214⟩, ⟨23204⟩⟩) exact67894RawTerms .large 67893 .exactZero (none)

def event67895 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25368⟩⟩) 0 ⟨23204⟩ 67894

def event67896 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25368⟩⟩) (.authority (.operator))

def exact67897RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨25368⟩⟩]⟩, (1)⟩]

theorem exact67897RawTermsValid :
    exact67897RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67897 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25368⟩⟩) exact67897RawTerms (.finite 8192) 67896 .exactZero (none)

def event67898 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨110⟩⟩) (.authority (.operator))

def event67899 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨110⟩⟩) .exactZero

def event67900 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12462⟩⟩) 0 ⟨12364⟩ 67886

def event67901 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12462⟩⟩) 1 ⟨110⟩ 67899

def event67902 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12462⟩⟩) (.sum [.predecessor 0 67900 .coefficient, .predecessor 1 67901 .coefficient])

def event67903 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12462⟩⟩) (.finite 1600)

def event67904 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12463⟩⟩) 0 ⟨12462⟩ 67903

def event67905 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12463⟩⟩) (.identity (.predecessor 0 67904 .coefficient))

def exact67906RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], []⟩, (1)⟩]

theorem exact67906RawTermsValid :
    exact67906RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67906 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12463⟩⟩) exact67906RawTerms (.finite 1600) 67905 .exactZero (none)

def event67907 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6544⟩⟩) (.authority (.factStore))

def exact67908RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact67908RawTermsValid :
    exact67908RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67908 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6544⟩⟩) exact67908RawTerms .large 67907 .exactZero (none)

def event67909 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12464⟩⟩) 0 ⟨6544⟩ 67908

def event67910 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12464⟩⟩) 1 ⟨12463⟩ 67906

def event67911 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12464⟩⟩) (.product (.predecessor 0 67909 .coefficient) (.predecessor 1 67910 .coefficient) (⟨false, false, none, none, none⟩))

def event67912 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12464⟩⟩, .operator (⟨67908, 0⟩, ⟨67906, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact67913RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact67913RawTermsValid :
    exact67913RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67913 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12464⟩⟩) exact67913RawTerms .large 67911 .exactZero (none)

def event67914 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.authority (.operator))

def event67915 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨2348⟩⟩) (.finite 1)

def event67916 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6757⟩⟩) 0 ⟨6689⟩ 67890

def event67917 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6757⟩⟩) (.authority (.operator))

def exact67918RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6757⟩⟩]⟩, (1)⟩]

theorem exact67918RawTermsValid :
    exact67918RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67918 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6757⟩⟩) exact67918RawTerms .large 67917 .exactZero (none)

def event67919 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6785⟩⟩) 0 ⟨6757⟩ 67918

def event67920 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6785⟩⟩) (.identity (.predecessor 0 67919 .coefficient))

def exact67921RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6785⟩⟩]⟩, (1)⟩]

theorem exact67921RawTermsValid :
    exact67921RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67921 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6785⟩⟩) exact67921RawTerms .large 67920 .exactZero (none)

def event67922 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7867⟩⟩) 0 ⟨6785⟩ 67921

def event67923 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7867⟩⟩) (.authority (.operator))

def exact67924RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩, (1)⟩]

theorem exact67924RawTermsValid :
    exact67924RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67924 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7867⟩⟩) exact67924RawTerms (.finite 8192) 67923 .exactZero (none)

def event67925 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7868⟩⟩) 0 ⟨7867⟩ 67924

def event67926 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7868⟩⟩) 1 ⟨2348⟩ 67915

def event67927 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7868⟩⟩) (.scale (.predecessor 0 67925 .coefficient) (.value (.predecessor 1 67926 .coefficient)))

def exact67928RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩, (1)⟩]

theorem exact67928RawTermsValid :
    exact67928RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67928 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7868⟩⟩) exact67928RawTerms (.finite 8192) 67927 .exactZero (none)

def event67929 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6765⟩⟩) 0 ⟨6757⟩ 67918

def event67930 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6765⟩⟩) (.identity (.predecessor 0 67929 .coefficient))

def exact67931RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6765⟩⟩]⟩, (1)⟩]

theorem exact67931RawTermsValid :
    exact67931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67931 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6765⟩⟩) exact67931RawTerms .large 67930 .exactZero (none)

def event67932 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7869⟩⟩) 0 ⟨6765⟩ 67931

def event67933 : Event := .predecessor (⟨.program ⟨214⟩, ⟨7869⟩⟩) 1 ⟨7868⟩ 67928

def event67934 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨7869⟩⟩) (.product (.predecessor 0 67932 .coefficient) (.predecessor 1 67933 .coefficient) (⟨false, false, none, none, none⟩))

def event67935 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨7869⟩⟩, .operator (⟨67931, 0⟩, ⟨67928, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩, (1)⟩)

def exact67936RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩, (1)⟩]

theorem exact67936RawTermsValid :
    exact67936RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67936 : Event := .resultExact (⟨.program ⟨214⟩, ⟨7869⟩⟩) exact67936RawTerms .large 67934 .exactZero (none)

def event67937 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12465⟩⟩) 0 ⟨7869⟩ 67936

def event67938 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12465⟩⟩) 1 ⟨12464⟩ 67913

def event67939 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12465⟩⟩) (.sum [.predecessor 0 67937 .coefficient, .predecessor 1 67938 .coefficient])

def exact67940RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact67940RawTermsValid :
    exact67940RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67940 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12465⟩⟩) exact67940RawTerms .large 67939 .exactZero (none)

def event67941 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25371⟩⟩) 0 ⟨12465⟩ 67940

def event67942 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25371⟩⟩) 1 ⟨25368⟩ 67897

def event67943 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25371⟩⟩) (.product (.predecessor 0 67941 .coefficient) (.predecessor 1 67942 .coefficient) (⟨false, false, none, none, none⟩))

def event67944 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25371⟩⟩, .operator (⟨67940, 0⟩, ⟨67897, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩]⟩, (1)⟩)

def event67945 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25371⟩⟩, .operator (⟨67940, 1⟩, ⟨67897, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩]⟩, (-1)⟩)

def event67946 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨25371⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25368⟩⟩) ⟨23204⟩ 67894)

def event67947 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25371⟩⟩, .relation 67946 0, ⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨23204⟩⟩]⟩, (-1)⟩)

def exact67948RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨23204⟩⟩]⟩, (-1)⟩]

theorem exact67948RawTermsValid :
    exact67948RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67948 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25371⟩⟩) exact67948RawTerms .large 67943 .exactZero (none)

def event67949 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16461⟩⟩) 0 ⟨12364⟩ 67886

def event67950 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16461⟩⟩) (.authority (.programFamilyFact))

def exact67951RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], []⟩, (1)⟩]

theorem exact67951RawTermsValid :
    exact67951RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67951 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16461⟩⟩) exact67951RawTerms (.finite 40) 67950 .exactZero (none)

def event67952 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16463⟩⟩) 0 ⟨6544⟩ 67908

def event67953 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16463⟩⟩) 1 ⟨16461⟩ 67951

def event67954 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16463⟩⟩) (.product (.predecessor 0 67952 .coefficient) (.predecessor 1 67953 .coefficient) (⟨false, true, none, none, some 1⟩))

def event67955 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨16463⟩⟩, .operator (⟨67908, 0⟩, ⟨67951, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩)

def exact67956RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (1)⟩]

theorem exact67956RawTermsValid :
    exact67956RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67956 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16463⟩⟩) exact67956RawTerms .large 67954 .exactZero (none)

def event67957 : Event := .predecessor (⟨.program ⟨214⟩, ⟨6702⟩⟩) 0 ⟨6689⟩ 67890

def event67958 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6702⟩⟩) (.authority (.operator))

def exact67959RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩]

theorem exact67959RawTermsValid :
    exact67959RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67959 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6702⟩⟩) exact67959RawTerms .large 67958 .exactZero (none)

def event67960 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16464⟩⟩) 0 ⟨6702⟩ 67959

def event67961 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16464⟩⟩) 1 ⟨16463⟩ 67956

def event67962 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16464⟩⟩) (.sum [.predecessor 0 67960 .coefficient, .predecessor 1 67961 .coefficient])

def exact67963RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact67963RawTermsValid :
    exact67963RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67963 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16464⟩⟩) exact67963RawTerms .large 67962 .exactZero (none)

def event67964 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25372⟩⟩) 0 ⟨16464⟩ 67963

def event67965 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25372⟩⟩) 1 ⟨25371⟩ 67948

def event67966 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25372⟩⟩) (.sum [.predecessor 0 67964 .coefficient, .predecessor 1 67965 .coefficient])

def exact67967RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨23204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact67967RawTermsValid :
    exact67967RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67967 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25372⟩⟩) exact67967RawTerms .large 67966 .exactZero (none)

def event67968 : Event := .preFoldPolynomial 67967 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨23204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩] .exactZero none

def exact67969RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨23204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

def event67969 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨25372⟩⟩) 67968 exact67969RawTerms .large 67966 .exactZero (none)

def event67970 : Event := .specializationComputed (⟨.program ⟨214⟩, ⟨12364⟩⟩) ⟨⟨115⟩, ⟨20⟩, ⟨109⟩⟩ ⟨67804, 67970⟩

def event67971 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨19887⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19884⟩⟩]⟩) (1) 0 2 (.universal 67970 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19884⟩⟩]⟩) (none) 67969)

def event67972 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19887⟩⟩, .relation 67971 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩)

def event67973 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19887⟩⟩, .relation 67971 1, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩]⟩, (-1)⟩)

def event67974 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19887⟩⟩, .relation 67971 2, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨23204⟩⟩]⟩, (1)⟩)

def event67975 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨19887⟩⟩, .relation 67971 3, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩)

def exact67976RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨23204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact67976RawTermsValid :
    exact67976RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67976 : Event := .resultExact (⟨.program ⟨214⟩, ⟨19887⟩⟩) exact67976RawTerms .large 67800 (.finite 1811303510016) (some (67802))

def event67977 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25370⟩⟩) 0 ⟨19887⟩ 67976

def event67978 : Event := .predecessor (⟨.program ⟨214⟩, ⟨25370⟩⟩) 1 ⟨25369⟩ 67790

def event67979 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25370⟩⟩) (.sum [.predecessor 0 67977 .coefficient, .predecessor 1 67978 .coefficient])

def event67980 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25370⟩⟩, .operator (⟨67976, 2⟩, ⟨67790, 1⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], [⟨.program ⟨214⟩, ⟨23204⟩⟩]⟩, (-1)⟩)

def event67981 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨25370⟩⟩, .operator (⟨67976, 1⟩, ⟨67790, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25368⟩⟩]⟩, (1)⟩)

def event67982 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨25370⟩⟩) (.sum [.result 67976 .summary, .result 67790 .summary])

def exact67983RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩, (-1)⟩]

theorem exact67983RawTermsValid :
    exact67983RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67983 : Event := .resultExact (⟨.program ⟨214⟩, ⟨25370⟩⟩) exact67983RawTerms .large 67979 (.finite 352127895089152) (some (67982))

def event67984 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28940⟩⟩) 0 ⟨25370⟩ 67983

def event67985 : Event := .predecessor (⟨.program ⟨214⟩, ⟨28940⟩⟩) 1 ⟨28938⟩ 67706

def event67986 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28940⟩⟩) (.product (.predecessor 0 67984 .coefficient) (.predecessor 1 67985 .coefficient) (⟨false, false, none, none, none⟩))

def event67987 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28940⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩) [⟨.result 67706 .coefficient, false, none⟩])

def event67988 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨28940⟩⟩) (.product (.result 67983 .summary) (.transfer 67987) (⟨false, false, none, none, none⟩))

def event67989 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28940⟩⟩, .operator (⟨67983, 0⟩, ⟨67706, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩, (1)⟩)

def event67990 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28940⟩⟩, .operator (⟨67983, 1⟩, ⟨67706, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩, (-1)⟩)

def event67991 : Event := .appliedRelation (⟨.program ⟨214⟩, ⟨28940⟩⟩) (⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28938⟩⟩) ⟨24474⟩ 67703)

def event67992 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨28940⟩⟩, .relation 67991 0, ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨24474⟩⟩]⟩, (-1)⟩)

def exact67993RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16461⟩⟩], [⟨.program ⟨214⟩, ⟨24474⟩⟩]⟩, (-1)⟩]

theorem exact67993RawTermsValid :
    exact67993RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67993 : Event := .resultExact (⟨.program ⟨214⟩, ⟨28940⟩⟩) exact67993RawTerms .large 67986 (.finite 1292315009023509266432) (some (67988))

def event67994 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22116⟩⟩) 0 ⟨16462⟩ 3218

def event67995 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22116⟩⟩) (.authority (.relationPreimageSource ⟨54⟩))

def exact67996RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22116⟩⟩]⟩, (1)⟩]

theorem exact67996RawTermsValid :
    exact67996RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event67996 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22116⟩⟩) exact67996RawTerms (.finite 136065468) 67995 .exactZero (none)

def event67997 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22118⟩⟩) 0 ⟨22116⟩ 67996

def event67998 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22118⟩⟩) 1 ⟨2348⟩ 4

def event67999 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22118⟩⟩) (.scale (.predecessor 0 67997 .coefficient) (.value (.predecessor 1 67998 .coefficient)))

def exact68000RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22116⟩⟩]⟩, (1)⟩]

theorem exact68000RawTermsValid :
    exact68000RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68000 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22118⟩⟩) exact68000RawTerms (.finite 136065468) 67999 .exactZero (none)

def event68001 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22119⟩⟩) 0 ⟨5535⟩ 65387

def event68002 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22119⟩⟩) 1 ⟨22118⟩ 68000

def event68003 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22119⟩⟩) (.product (.predecessor 0 68001 .coefficient) (.predecessor 1 68002 .coefficient) (⟨false, false, none, none, none⟩))

def event68004 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22119⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨214⟩, ⟨22116⟩⟩]⟩) [⟨.result 67996 .coefficient, false, none⟩])

def event68005 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22119⟩⟩) (.product (.result 65387 .summary) (.transfer 68004) (⟨false, false, none, none, none⟩))

def event68006 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22119⟩⟩, .operator (⟨65387, 0⟩, ⟨68000, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22116⟩⟩]⟩, (1)⟩)

def event68007 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨22117⟩⟩)

def event68008 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event68009 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event68010 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event68011 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event68012 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event68013 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event68014 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event68015 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event68016 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 68015

def event68017 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 68013

def event68018 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 68016 .coefficient) (.value (.predecessor 1 68017 .coefficient)))

def event68019 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event68020 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 68019

def event68021 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 68011

def event68022 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 68020 .coefficient, .predecessor 1 68021 .coefficient])

def event68023 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event68024 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 68023

def event68025 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 68009

def event68026 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 68025 .coefficient))

def event68027 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event68028 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12362⟩⟩) 0 ⟨5530⟩ 68027

def event68029 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12362⟩⟩) (.authority (.programFamilyFact))

def exact68030RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12362⟩⟩], []⟩, (1)⟩]

theorem exact68030RawTermsValid :
    exact68030RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68030 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12362⟩⟩) exact68030RawTerms (.finite 40) 68029 .exactZero (none)

def event68031 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9815⟩⟩) 0 ⟨5530⟩ 68027

def event68032 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9815⟩⟩) (.authority (.programFamilyFact))

def exact68033RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩], []⟩, (1)⟩]

theorem exact68033RawTermsValid :
    exact68033RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68033 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9815⟩⟩) exact68033RawTerms (.finite 40) 68032 .exactZero (none)

def event68034 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12363⟩⟩) 0 ⟨9815⟩ 68033

def event68035 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12363⟩⟩) 1 ⟨12362⟩ 68030

def event68036 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12363⟩⟩) (.product (.predecessor 0 68034 .coefficient) (.predecessor 1 68035 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event68037 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12363⟩⟩) (.monomialProduct (⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], []⟩) [⟨.result 68033 .coefficient, true, some 1⟩, ⟨.result 68030 .coefficient, true, some 1⟩])

def event68038 : Event := .survivorFold (1) 68037

def exact68039RawTerms : List Term := []

theorem exact68039RawTermsValid :
    exact68039RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68039 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12363⟩⟩) exact68039RawTerms (.finite 1600) 68036 (.finite 1600) (some (68037))

def event68040 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12364⟩⟩) 0 ⟨12363⟩ 68039

def event68041 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12364⟩⟩) (.identity (.predecessor 0 68040 .coefficient))

def event68042 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12364⟩⟩) (.finite 1600)

def event68043 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16461⟩⟩) 0 ⟨12364⟩ 68042

def event68044 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16461⟩⟩) (.authority (.programFamilyFact))

def exact68045RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨16461⟩⟩], []⟩, (1)⟩]

theorem exact68045RawTermsValid :
    exact68045RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68045 : Event := .resultExact (⟨.program ⟨214⟩, ⟨16461⟩⟩) exact68045RawTerms (.finite 40) 68044 .exactZero (none)

def event68046 : Event := .predecessor (⟨.program ⟨214⟩, ⟨16462⟩⟩) 0 ⟨16461⟩ 68045

def event68047 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨16462⟩⟩) (.identity (.predecessor 0 68046 .coefficient))

def event68048 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨16462⟩⟩) (.finite 40)

def event68049 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22116⟩⟩) 0 ⟨16462⟩ 68048

def event68050 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22116⟩⟩) (.authority (.relationPreimageSource ⟨54⟩))

def exact68051RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨22116⟩⟩]⟩, (1)⟩]

theorem exact68051RawTermsValid :
    exact68051RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68051 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22116⟩⟩) exact68051RawTerms (.finite 136065468) 68050 .exactZero (none)

def event68052 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨6⟩⟩) (.authority (.operator))

def exact68053RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩, (1)⟩]

theorem exact68053RawTermsValid :
    exact68053RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68053 : Event := .resultExact (⟨.program ⟨214⟩, ⟨6⟩⟩) exact68053RawTerms .large 68052 .exactZero (none)

def event68054 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22117⟩⟩) 0 ⟨6⟩ 68053

def event68055 : Event := .predecessor (⟨.program ⟨214⟩, ⟨22117⟩⟩) 1 ⟨22116⟩ 68051

def event68056 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨22117⟩⟩) (.product (.predecessor 0 68054 .coefficient) (.predecessor 1 68055 .coefficient) (⟨false, false, none, none, none⟩))

def event68057 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨22117⟩⟩, .operator (⟨68053, 0⟩, ⟨68051, 0⟩), ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22116⟩⟩]⟩, (1)⟩)

def exact68058RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22116⟩⟩]⟩, (1)⟩]

theorem exact68058RawTermsValid :
    exact68058RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68058 : Event := .resultExact (⟨.program ⟨214⟩, ⟨22117⟩⟩) exact68058RawTerms .large 68056 .exactZero (none)

def event68059 : Event := .preFoldPolynomial 68058 [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22116⟩⟩]⟩, (1)⟩] .exactZero none

def exact68060RawTerms : List Term := [⟨⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22116⟩⟩]⟩, (1)⟩]

def event68060 : Event := .invocationEndExact (⟨.program ⟨214⟩, ⟨22117⟩⟩) 68059 exact68060RawTerms .large 68056 .exactZero (none)

def event68061 : Event := .invocationStart (⟨.program ⟨214⟩, ⟨28943⟩⟩)

def event68062 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨961⟩⟩) (.authority (.operator))

def event68063 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨961⟩⟩) (.finite 224)

def event68064 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨360⟩⟩) (.authority (.operator))

def event68065 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨360⟩⟩) (.finite 2)

def event68066 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.authority (.operator))

def event68067 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5501⟩⟩) (.finite 7)

def event68068 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨71⟩⟩) (.authority (.operator))

def event68069 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨71⟩⟩) (.finite 31)

def event68070 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 0 ⟨71⟩ 68069

def event68071 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5503⟩⟩) 1 ⟨5501⟩ 68067

def event68072 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.scale (.predecessor 0 68070 .coefficient) (.value (.predecessor 1 68071 .coefficient)))

def event68073 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5503⟩⟩) (.finite 217)

def event68074 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 0 ⟨5503⟩ 68073

def event68075 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5504⟩⟩) 1 ⟨360⟩ 68065

def event68076 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.sum [.predecessor 0 68074 .coefficient, .predecessor 1 68075 .coefficient])

def event68077 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5504⟩⟩) (.finite 219)

def event68078 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 0 ⟨5504⟩ 68077

def event68079 : Event := .predecessor (⟨.program ⟨214⟩, ⟨5530⟩⟩) 1 ⟨961⟩ 68063

def event68080 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.identity (.predecessor 1 68079 .coefficient))

def event68081 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨5530⟩⟩) (.finite 224)

def event68082 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12362⟩⟩) 0 ⟨5530⟩ 68081

def event68083 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12362⟩⟩) (.authority (.programFamilyFact))

def exact68084RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨12362⟩⟩], []⟩, (1)⟩]

theorem exact68084RawTermsValid :
    exact68084RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68084 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12362⟩⟩) exact68084RawTerms (.finite 40) 68083 .exactZero (none)

def event68085 : Event := .predecessor (⟨.program ⟨214⟩, ⟨9815⟩⟩) 0 ⟨5530⟩ 68081

def event68086 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨9815⟩⟩) (.authority (.programFamilyFact))

def exact68087RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩], []⟩, (1)⟩]

theorem exact68087RawTermsValid :
    exact68087RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68087 : Event := .resultExact (⟨.program ⟨214⟩, ⟨9815⟩⟩) exact68087RawTerms (.finite 40) 68086 .exactZero (none)

def event68088 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12363⟩⟩) 0 ⟨9815⟩ 68087

def event68089 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12363⟩⟩) 1 ⟨12362⟩ 68084

def event68090 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12363⟩⟩) (.product (.predecessor 0 68088 .coefficient) (.predecessor 1 68089 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event68091 : Event := .coefficientMerge (⟨⟨.program ⟨214⟩, ⟨12363⟩⟩, .operator (⟨68087, 0⟩, ⟨68084, 0⟩), ⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], []⟩, (1)⟩)

def exact68092RawTerms : List Term := [⟨⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩, ⟨.program ⟨214⟩, ⟨12362⟩⟩], []⟩, (1)⟩]

theorem exact68092RawTermsValid :
    exact68092RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event68092 : Event := .resultExact (⟨.program ⟨214⟩, ⟨12363⟩⟩) exact68092RawTerms (.finite 1600) 68090 .exactZero (none)

def event68093 : Event := .predecessor (⟨.program ⟨214⟩, ⟨12364⟩⟩) 0 ⟨12363⟩ 68092

def event68094 : Event := .boundTransfer (⟨.program ⟨214⟩, ⟨12364⟩⟩) (.identity (.predecessor 0 68093 .coefficient))

def event68095 : Event := .resultCoefficient (⟨.program ⟨214⟩, ⟨12364⟩⟩) (.finite 1600)

def eventLeaf4240 : Array AnnotatedEvent := #[
  { event := event67840
    frameStart := 67804 },
  { event := event67841
    frameStart := 67804 },
  { event := event67842
    frameStart := 67804 },
  { event := event67843
    frameStart := 67804 },
  { event := event67844
    frameStart := 67804 },
  { event := event67845
    frameStart := 67804 },
  { event := event67846
    frameStart := 67804 },
  { event := event67847
    frameStart := 67804 },
  { event := event67848
    frameStart := 67804 },
  { event := event67849
    frameStart := 67804 },
  { event := event67850
    frameStart := 67804 },
  { event := event67851
    frameStart := 67804 },
  { event := event67852
    frameStart := 67852 },
  { event := event67853
    frameStart := 67852 },
  { event := event67854
    frameStart := 67852 },
  { event := event67855
    frameStart := 67852 }
]

def eventLeaf4241 : Array AnnotatedEvent := #[
  { event := event67856
    frameStart := 67852 },
  { event := event67857
    frameStart := 67852 },
  { event := event67858
    frameStart := 67852 },
  { event := event67859
    frameStart := 67852 },
  { event := event67860
    frameStart := 67852 },
  { event := event67861
    frameStart := 67852 },
  { event := event67862
    frameStart := 67852 },
  { event := event67863
    frameStart := 67852 },
  { event := event67864
    frameStart := 67852 },
  { event := event67865
    frameStart := 67852 },
  { event := event67866
    frameStart := 67852 },
  { event := event67867
    frameStart := 67852 },
  { event := event67868
    frameStart := 67852 },
  { event := event67869
    frameStart := 67852 },
  { event := event67870
    frameStart := 67852 },
  { event := event67871
    frameStart := 67852 }
]

def eventLeaf4242 : Array AnnotatedEvent := #[
  { event := event67872
    frameStart := 67852 },
  { event := event67873
    frameStart := 67852 },
  { event := event67874
    frameStart := 67852 },
  { event := event67875
    frameStart := 67852 },
  { event := event67876
    frameStart := 67852 },
  { event := event67877
    frameStart := 67852 },
  { event := event67878
    frameStart := 67852 },
  { event := event67879
    frameStart := 67852 },
  { event := event67880
    frameStart := 67852 },
  { event := event67881
    frameStart := 67852 },
  { event := event67882
    frameStart := 67852 },
  { event := event67883
    frameStart := 67852 },
  { event := event67884
    frameStart := 67852 },
  { event := event67885
    frameStart := 67852 },
  { event := event67886
    frameStart := 67852 },
  { event := event67887
    frameStart := 67852 }
]

def eventLeaf4243 : Array AnnotatedEvent := #[
  { event := event67888
    frameStart := 67852 },
  { event := event67889
    frameStart := 67852 },
  { event := event67890
    frameStart := 67852 },
  { event := event67891
    frameStart := 67852 },
  { event := event67892
    frameStart := 67852 },
  { event := event67893
    frameStart := 67852 },
  { event := event67894
    frameStart := 67852 },
  { event := event67895
    frameStart := 67852 },
  { event := event67896
    frameStart := 67852 },
  { event := event67897
    frameStart := 67852 },
  { event := event67898
    frameStart := 67852 },
  { event := event67899
    frameStart := 67852 },
  { event := event67900
    frameStart := 67852 },
  { event := event67901
    frameStart := 67852 },
  { event := event67902
    frameStart := 67852 },
  { event := event67903
    frameStart := 67852 }
]

def eventLeaf4244 : Array AnnotatedEvent := #[
  { event := event67904
    frameStart := 67852 },
  { event := event67905
    frameStart := 67852 },
  { event := event67906
    frameStart := 67852 },
  { event := event67907
    frameStart := 67852 },
  { event := event67908
    frameStart := 67852 },
  { event := event67909
    frameStart := 67852 },
  { event := event67910
    frameStart := 67852 },
  { event := event67911
    frameStart := 67852 },
  { event := event67912
    frameStart := 67852 },
  { event := event67913
    frameStart := 67852 },
  { event := event67914
    frameStart := 67852 },
  { event := event67915
    frameStart := 67852 },
  { event := event67916
    frameStart := 67852 },
  { event := event67917
    frameStart := 67852 },
  { event := event67918
    frameStart := 67852 },
  { event := event67919
    frameStart := 67852 }
]

def eventLeaf4245 : Array AnnotatedEvent := #[
  { event := event67920
    frameStart := 67852 },
  { event := event67921
    frameStart := 67852 },
  { event := event67922
    frameStart := 67852 },
  { event := event67923
    frameStart := 67852 },
  { event := event67924
    frameStart := 67852 },
  { event := event67925
    frameStart := 67852 },
  { event := event67926
    frameStart := 67852 },
  { event := event67927
    frameStart := 67852 },
  { event := event67928
    frameStart := 67852 },
  { event := event67929
    frameStart := 67852 },
  { event := event67930
    frameStart := 67852 },
  { event := event67931
    frameStart := 67852 },
  { event := event67932
    frameStart := 67852 },
  { event := event67933
    frameStart := 67852 },
  { event := event67934
    frameStart := 67852 },
  { event := event67935
    frameStart := 67852 }
]

def eventLeaf4246 : Array AnnotatedEvent := #[
  { event := event67936
    frameStart := 67852 },
  { event := event67937
    frameStart := 67852 },
  { event := event67938
    frameStart := 67852 },
  { event := event67939
    frameStart := 67852 },
  { event := event67940
    frameStart := 67852 },
  { event := event67941
    frameStart := 67852 },
  { event := event67942
    frameStart := 67852 },
  { event := event67943
    frameStart := 67852 },
  { event := event67944
    frameStart := 67852 },
  { event := event67945
    frameStart := 67852 },
  { event := event67946
    frameStart := 67852 },
  { event := event67947
    frameStart := 67852 },
  { event := event67948
    frameStart := 67852 },
  { event := event67949
    frameStart := 67852 },
  { event := event67950
    frameStart := 67852 },
  { event := event67951
    frameStart := 67852 }
]

def eventLeaf4247 : Array AnnotatedEvent := #[
  { event := event67952
    frameStart := 67852 },
  { event := event67953
    frameStart := 67852 },
  { event := event67954
    frameStart := 67852 },
  { event := event67955
    frameStart := 67852 },
  { event := event67956
    frameStart := 67852 },
  { event := event67957
    frameStart := 67852 },
  { event := event67958
    frameStart := 67852 },
  { event := event67959
    frameStart := 67852 },
  { event := event67960
    frameStart := 67852 },
  { event := event67961
    frameStart := 67852 },
  { event := event67962
    frameStart := 67852 },
  { event := event67963
    frameStart := 67852 },
  { event := event67964
    frameStart := 67852 },
  { event := event67965
    frameStart := 67852 },
  { event := event67966
    frameStart := 67852 },
  { event := event67967
    frameStart := 67852 }
]

def eventLeaf4248 : Array AnnotatedEvent := #[
  { event := event67968
    frameStart := 67852 },
  { event := event67969
    frameStart := 67852 },
  { event := event67970
    frameStart := 0 },
  { event := event67971
    frameStart := 0 },
  { event := event67972
    frameStart := 0 },
  { event := event67973
    frameStart := 0 },
  { event := event67974
    frameStart := 0 },
  { event := event67975
    frameStart := 0 },
  { event := event67976
    frameStart := 0 },
  { event := event67977
    frameStart := 0 },
  { event := event67978
    frameStart := 0 },
  { event := event67979
    frameStart := 0 },
  { event := event67980
    frameStart := 0 },
  { event := event67981
    frameStart := 0 },
  { event := event67982
    frameStart := 0 },
  { event := event67983
    frameStart := 0 }
]

def eventLeaf4249 : Array AnnotatedEvent := #[
  { event := event67984
    frameStart := 0 },
  { event := event67985
    frameStart := 0 },
  { event := event67986
    frameStart := 0 },
  { event := event67987
    frameStart := 0 },
  { event := event67988
    frameStart := 0 },
  { event := event67989
    frameStart := 0 },
  { event := event67990
    frameStart := 0 },
  { event := event67991
    frameStart := 0 },
  { event := event67992
    frameStart := 0 },
  { event := event67993
    frameStart := 0 },
  { event := event67994
    frameStart := 0 },
  { event := event67995
    frameStart := 0 },
  { event := event67996
    frameStart := 0 },
  { event := event67997
    frameStart := 0 },
  { event := event67998
    frameStart := 0 },
  { event := event67999
    frameStart := 0 }
]

def eventLeaf4250 : Array AnnotatedEvent := #[
  { event := event68000
    frameStart := 0 },
  { event := event68001
    frameStart := 0 },
  { event := event68002
    frameStart := 0 },
  { event := event68003
    frameStart := 0 },
  { event := event68004
    frameStart := 0 },
  { event := event68005
    frameStart := 0 },
  { event := event68006
    frameStart := 0 },
  { event := event68007
    frameStart := 68007 },
  { event := event68008
    frameStart := 68007 },
  { event := event68009
    frameStart := 68007 },
  { event := event68010
    frameStart := 68007 },
  { event := event68011
    frameStart := 68007 },
  { event := event68012
    frameStart := 68007 },
  { event := event68013
    frameStart := 68007 },
  { event := event68014
    frameStart := 68007 },
  { event := event68015
    frameStart := 68007 }
]

def eventLeaf4251 : Array AnnotatedEvent := #[
  { event := event68016
    frameStart := 68007 },
  { event := event68017
    frameStart := 68007 },
  { event := event68018
    frameStart := 68007 },
  { event := event68019
    frameStart := 68007 },
  { event := event68020
    frameStart := 68007 },
  { event := event68021
    frameStart := 68007 },
  { event := event68022
    frameStart := 68007 },
  { event := event68023
    frameStart := 68007 },
  { event := event68024
    frameStart := 68007 },
  { event := event68025
    frameStart := 68007 },
  { event := event68026
    frameStart := 68007 },
  { event := event68027
    frameStart := 68007 },
  { event := event68028
    frameStart := 68007 },
  { event := event68029
    frameStart := 68007 },
  { event := event68030
    frameStart := 68007 },
  { event := event68031
    frameStart := 68007 }
]

def eventLeaf4252 : Array AnnotatedEvent := #[
  { event := event68032
    frameStart := 68007 },
  { event := event68033
    frameStart := 68007 },
  { event := event68034
    frameStart := 68007 },
  { event := event68035
    frameStart := 68007 },
  { event := event68036
    frameStart := 68007 },
  { event := event68037
    frameStart := 68007 },
  { event := event68038
    frameStart := 68007 },
  { event := event68039
    frameStart := 68007 },
  { event := event68040
    frameStart := 68007 },
  { event := event68041
    frameStart := 68007 },
  { event := event68042
    frameStart := 68007 },
  { event := event68043
    frameStart := 68007 },
  { event := event68044
    frameStart := 68007 },
  { event := event68045
    frameStart := 68007 },
  { event := event68046
    frameStart := 68007 },
  { event := event68047
    frameStart := 68007 }
]

def eventLeaf4253 : Array AnnotatedEvent := #[
  { event := event68048
    frameStart := 68007 },
  { event := event68049
    frameStart := 68007 },
  { event := event68050
    frameStart := 68007 },
  { event := event68051
    frameStart := 68007 },
  { event := event68052
    frameStart := 68007 },
  { event := event68053
    frameStart := 68007 },
  { event := event68054
    frameStart := 68007 },
  { event := event68055
    frameStart := 68007 },
  { event := event68056
    frameStart := 68007 },
  { event := event68057
    frameStart := 68007 },
  { event := event68058
    frameStart := 68007 },
  { event := event68059
    frameStart := 68007 },
  { event := event68060
    frameStart := 68007 },
  { event := event68061
    frameStart := 68061 },
  { event := event68062
    frameStart := 68061 },
  { event := event68063
    frameStart := 68061 }
]

def eventLeaf4254 : Array AnnotatedEvent := #[
  { event := event68064
    frameStart := 68061 },
  { event := event68065
    frameStart := 68061 },
  { event := event68066
    frameStart := 68061 },
  { event := event68067
    frameStart := 68061 },
  { event := event68068
    frameStart := 68061 },
  { event := event68069
    frameStart := 68061 },
  { event := event68070
    frameStart := 68061 },
  { event := event68071
    frameStart := 68061 },
  { event := event68072
    frameStart := 68061 },
  { event := event68073
    frameStart := 68061 },
  { event := event68074
    frameStart := 68061 },
  { event := event68075
    frameStart := 68061 },
  { event := event68076
    frameStart := 68061 },
  { event := event68077
    frameStart := 68061 },
  { event := event68078
    frameStart := 68061 },
  { event := event68079
    frameStart := 68061 }
]

def eventLeaf4255 : Array AnnotatedEvent := #[
  { event := event68080
    frameStart := 68061 },
  { event := event68081
    frameStart := 68061 },
  { event := event68082
    frameStart := 68061 },
  { event := event68083
    frameStart := 68061 },
  { event := event68084
    frameStart := 68061 },
  { event := event68085
    frameStart := 68061 },
  { event := event68086
    frameStart := 68061 },
  { event := event68087
    frameStart := 68061 },
  { event := event68088
    frameStart := 68061 },
  { event := event68089
    frameStart := 68061 },
  { event := event68090
    frameStart := 68061 },
  { event := event68091
    frameStart := 68061 },
  { event := event68092
    frameStart := 68061 },
  { event := event68093
    frameStart := 68061 },
  { event := event68094
    frameStart := 68061 },
  { event := event68095
    frameStart := 68061 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.Events265
