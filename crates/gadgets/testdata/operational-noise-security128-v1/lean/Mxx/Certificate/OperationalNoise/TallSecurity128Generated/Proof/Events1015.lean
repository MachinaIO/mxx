import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1015

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact259840RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], [⟨.program ⟨257⟩, ⟨16819⟩⟩]⟩, (-1)⟩]

theorem exact259840RawTermsValid :
    exact259840RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259840 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17307⟩⟩) exact259840RawTerms .large 259835 .exactZero (none)

def event259841 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15748⟩⟩) 0 ⟨15356⟩ 259778

def event259842 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15748⟩⟩) (.authority (.programFamilyFact))

def exact259843RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], []⟩, (1)⟩]

theorem exact259843RawTermsValid :
    exact259843RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259843 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15748⟩⟩) exact259843RawTerms (.finite 2) 259842 .exactZero (none)

def event259844 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15750⟩⟩) 0 ⟨6908⟩ 259800

def event259845 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15750⟩⟩) 1 ⟨15748⟩ 259843

def event259846 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15750⟩⟩) (.product (.predecessor 0 259844 .coefficient) (.predecessor 1 259845 .coefficient) (⟨false, true, none, none, some 1⟩))

def event259847 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15750⟩⟩, .operator (⟨259800, 0⟩, ⟨259843, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact259848RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact259848RawTermsValid :
    exact259848RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259848 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15750⟩⟩) exact259848RawTerms .large 259846 .exactZero (none)

def event259849 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 259782

def event259850 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact259851RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact259851RawTermsValid :
    exact259851RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259851 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact259851RawTerms .large 259850 .exactZero (none)

def event259852 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15751⟩⟩) 0 ⟨7179⟩ 259851

def event259853 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15751⟩⟩) 1 ⟨15750⟩ 259848

def event259854 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15751⟩⟩) (.sum [.predecessor 0 259852 .coefficient, .predecessor 1 259853 .coefficient])

def exact259855RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact259855RawTermsValid :
    exact259855RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259855 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15751⟩⟩) exact259855RawTerms .large 259854 .exactZero (none)

def event259856 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17308⟩⟩) 0 ⟨15751⟩ 259855

def event259857 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17308⟩⟩) 1 ⟨17307⟩ 259840

def event259858 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17308⟩⟩) (.sum [.predecessor 0 259856 .coefficient, .predecessor 1 259857 .coefficient])

def exact259859RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], [⟨.program ⟨257⟩, ⟨16819⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact259859RawTermsValid :
    exact259859RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259859 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17308⟩⟩) exact259859RawTerms .large 259858 .exactZero (none)

def event259860 : Event := .preFoldPolynomial 259859 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], [⟨.program ⟨257⟩, ⟨16819⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact259861RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], [⟨.program ⟨257⟩, ⟨16819⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event259861 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17308⟩⟩) 259860 exact259861RawTerms .large 259858 .exactZero (none)

def event259862 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15356⟩⟩) ⟨⟨58⟩, ⟨36⟩, ⟨135⟩⟩ ⟨259696, 259862⟩

def event259863 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16242⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16239⟩⟩]⟩) (1) 0 2 (.universal 259862 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16239⟩⟩]⟩) (none) 259861)

def event259864 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16242⟩⟩, .relation 259863 0, ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩)

def event259865 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16242⟩⟩, .relation 259863 1, ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩]⟩, (-1)⟩)

def event259866 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16242⟩⟩, .relation 259863 2, ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], [⟨.program ⟨257⟩, ⟨16819⟩⟩]⟩, (1)⟩)

def event259867 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16242⟩⟩, .relation 259863 3, ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact259868RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], [⟨.program ⟨257⟩, ⟨16819⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact259868RawTermsValid :
    exact259868RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259868 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16242⟩⟩) exact259868RawTerms .large 259692 (.finite 202072841853861888) (some (259694))

def event259869 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17306⟩⟩) 0 ⟨16242⟩ 259868

def event259870 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17306⟩⟩) 1 ⟨17305⟩ 259682

def event259871 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17306⟩⟩) (.sum [.predecessor 0 259869 .coefficient, .predecessor 1 259870 .coefficient])

def event259872 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17306⟩⟩, .operator (⟨259868, 2⟩, ⟨259682, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], [⟨.program ⟨257⟩, ⟨16819⟩⟩]⟩, (-1)⟩)

def event259873 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17306⟩⟩, .operator (⟨259868, 1⟩, ⟨259682, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩]⟩, (1)⟩)

def event259874 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17306⟩⟩) (.sum [.result 259868 .summary, .result 259682 .summary])

def exact259875RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact259875RawTermsValid :
    exact259875RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259875 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17306⟩⟩) exact259875RawTerms .large 259871 (.finite 2997816280693142192128) (some (259874))

def event259876 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17623⟩⟩) 0 ⟨17306⟩ 259875

def event259877 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17623⟩⟩) 1 ⟨17621⟩ 259598

def event259878 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17623⟩⟩) (.product (.predecessor 0 259876 .coefficient) (.predecessor 1 259877 .coefficient) (⟨false, false, none, none, none⟩))

def event259879 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17623⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩) [⟨.result 259598 .coefficient, false, none⟩])

def event259880 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17623⟩⟩) (.product (.result 259875 .summary) (.transfer 259879) (⟨false, false, none, none, none⟩))

def event259881 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17623⟩⟩, .operator (⟨259875, 0⟩, ⟨259598, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩, (1)⟩)

def event259882 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17623⟩⟩, .operator (⟨259875, 1⟩, ⟨259598, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩, (-1)⟩)

def event259883 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17623⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17621⟩⟩) ⟨16956⟩ 259595)

def event259884 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17623⟩⟩, .relation 259883 0, ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨16956⟩⟩]⟩, (-1)⟩)

def exact259885RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨16956⟩⟩]⟩, (-1)⟩]

theorem exact259885RawTermsValid :
    exact259885RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259885 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17623⟩⟩) exact259885RawTerms .large 259878 (.finite 32188807212483504816668771614720) (some (259880))

def event259886 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16496⟩⟩) 0 ⟨15749⟩ 12470

def event259887 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16496⟩⟩) (.authority (.relationPreimageSource ⟨57⟩))

def exact259888RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16496⟩⟩]⟩, (1)⟩]

theorem exact259888RawTermsValid :
    exact259888RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259888 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16496⟩⟩) exact259888RawTerms (.finite 5647228698) 259887 .exactZero (none)

def event259889 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16498⟩⟩) 0 ⟨16496⟩ 259888

def event259890 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16498⟩⟩) 1 ⟨2370⟩ 4

def event259891 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16498⟩⟩) (.scale (.predecessor 0 259889 .coefficient) (.value (.predecessor 1 259890 .coefficient)))

def exact259892RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16496⟩⟩]⟩, (1)⟩]

theorem exact259892RawTermsValid :
    exact259892RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259892 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16498⟩⟩) exact259892RawTerms (.finite 5647228698) 259891 .exactZero (none)

def event259893 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16499⟩⟩) 0 ⟨5509⟩ 251495

def event259894 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16499⟩⟩) 1 ⟨16498⟩ 259892

def event259895 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16499⟩⟩) (.product (.predecessor 0 259893 .coefficient) (.predecessor 1 259894 .coefficient) (⟨false, false, none, none, none⟩))

def event259896 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16499⟩⟩) (.monomialProduct (⟨[], [⟨.program ⟨257⟩, ⟨16496⟩⟩]⟩) [⟨.result 259888 .coefficient, false, none⟩])

def event259897 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16499⟩⟩) (.product (.result 251495 .summary) (.transfer 259896) (⟨false, false, none, none, none⟩))

def event259898 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16499⟩⟩, .operator (⟨251495, 0⟩, ⟨259892, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16496⟩⟩]⟩, (1)⟩)

def event259899 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨16497⟩⟩)

def event259900 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event259901 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event259902 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2880⟩⟩) (.authority (.operator))

def event259903 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2880⟩⟩) (.finite 3)

def event259904 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event259905 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event259906 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event259907 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event259908 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 259907

def event259909 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 259905

def event259910 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 259908 .coefficient) (.value (.predecessor 1 259909 .coefficient)))

def event259911 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event259912 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2882⟩⟩) 0 ⟨392⟩ 259911

def event259913 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2882⟩⟩) 1 ⟨2880⟩ 259903

def event259914 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2882⟩⟩) (.sum [.predecessor 0 259912 .coefficient, .predecessor 1 259913 .coefficient])

def event259915 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2882⟩⟩) (.finite 655343)

def event259916 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5505⟩⟩) 0 ⟨2882⟩ 259915

def event259917 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5505⟩⟩) 1 ⟨5426⟩ 259901

def event259918 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5505⟩⟩) (.identity (.predecessor 1 259917 .coefficient))

def event259919 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5505⟩⟩) (.finite 655360)

def event259920 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15354⟩⟩) 0 ⟨5505⟩ 259919

def event259921 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15354⟩⟩) (.authority (.programFamilyFact))

def exact259922RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15354⟩⟩], []⟩, (1)⟩]

theorem exact259922RawTermsValid :
    exact259922RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259922 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15354⟩⟩) exact259922RawTerms (.finite 2) 259921 .exactZero (none)

def event259923 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12306⟩⟩) 0 ⟨5505⟩ 259919

def event259924 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12306⟩⟩) (.authority (.programFamilyFact))

def exact259925RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12306⟩⟩], []⟩, (1)⟩]

theorem exact259925RawTermsValid :
    exact259925RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259925 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12306⟩⟩) exact259925RawTerms (.finite 2) 259924 .exactZero (none)

def event259926 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15355⟩⟩) 0 ⟨12306⟩ 259925

def event259927 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15355⟩⟩) 1 ⟨15354⟩ 259922

def event259928 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15355⟩⟩) (.product (.predecessor 0 259926 .coefficient) (.predecessor 1 259927 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event259929 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15355⟩⟩) (.monomialProduct (⟨[⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], []⟩) [⟨.result 259925 .coefficient, true, some 1⟩, ⟨.result 259922 .coefficient, true, some 1⟩])

def event259930 : Event := .survivorFold (1) 259929

def exact259931RawTerms : List Term := []

theorem exact259931RawTermsValid :
    exact259931RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259931 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15355⟩⟩) exact259931RawTerms (.finite 4) 259928 (.finite 4) (some (259929))

def event259932 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15356⟩⟩) 0 ⟨15355⟩ 259931

def event259933 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15356⟩⟩) (.identity (.predecessor 0 259932 .coefficient))

def event259934 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15356⟩⟩) (.finite 4)

def event259935 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15748⟩⟩) 0 ⟨15356⟩ 259934

def event259936 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15748⟩⟩) (.authority (.programFamilyFact))

def exact259937RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], []⟩, (1)⟩]

theorem exact259937RawTermsValid :
    exact259937RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259937 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15748⟩⟩) exact259937RawTerms (.finite 2) 259936 .exactZero (none)

def event259938 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15749⟩⟩) 0 ⟨15748⟩ 259937

def event259939 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15749⟩⟩) (.identity (.predecessor 0 259938 .coefficient))

def event259940 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15749⟩⟩) (.finite 2)

def event259941 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16496⟩⟩) 0 ⟨15749⟩ 259940

def event259942 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16496⟩⟩) (.authority (.relationPreimageSource ⟨57⟩))

def exact259943RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16496⟩⟩]⟩, (1)⟩]

theorem exact259943RawTermsValid :
    exact259943RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259943 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16496⟩⟩) exact259943RawTerms (.finite 5647228698) 259942 .exactZero (none)

def event259944 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35⟩⟩) (.authority (.operator))

def exact259945RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩, (1)⟩]

theorem exact259945RawTermsValid :
    exact259945RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259945 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35⟩⟩) exact259945RawTerms .large 259944 .exactZero (none)

def event259946 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16497⟩⟩) 0 ⟨35⟩ 259945

def event259947 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16497⟩⟩) 1 ⟨16496⟩ 259943

def event259948 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16497⟩⟩) (.product (.predecessor 0 259946 .coefficient) (.predecessor 1 259947 .coefficient) (⟨false, false, none, none, none⟩))

def event259949 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16497⟩⟩, .operator (⟨259945, 0⟩, ⟨259943, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16496⟩⟩]⟩, (1)⟩)

def exact259950RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16496⟩⟩]⟩, (1)⟩]

theorem exact259950RawTermsValid :
    exact259950RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259950 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16497⟩⟩) exact259950RawTerms .large 259948 .exactZero (none)

def event259951 : Event := .preFoldPolynomial 259950 [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16496⟩⟩]⟩, (1)⟩] .exactZero none

def exact259952RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16496⟩⟩]⟩, (1)⟩]

def event259952 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨16497⟩⟩) 259951 exact259952RawTerms .large 259948 .exactZero (none)

def event259953 : Event := .invocationStart (⟨.program ⟨257⟩, ⟨17625⟩⟩)

def event259954 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.authority (.operator))

def event259955 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5426⟩⟩) (.finite 655360)

def event259956 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2880⟩⟩) (.authority (.operator))

def event259957 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2880⟩⟩) (.finite 3)

def event259958 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨390⟩⟩) (.authority (.operator))

def event259959 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨390⟩⟩) (.finite 20)

def event259960 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨97⟩⟩) (.authority (.operator))

def event259961 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨97⟩⟩) (.finite 32767)

def event259962 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 0 ⟨97⟩ 259961

def event259963 : Event := .predecessor (⟨.program ⟨257⟩, ⟨392⟩⟩) 1 ⟨390⟩ 259959

def event259964 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨392⟩⟩) (.scale (.predecessor 0 259962 .coefficient) (.value (.predecessor 1 259963 .coefficient)))

def event259965 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨392⟩⟩) (.finite 655340)

def event259966 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2882⟩⟩) 0 ⟨392⟩ 259965

def event259967 : Event := .predecessor (⟨.program ⟨257⟩, ⟨2882⟩⟩) 1 ⟨2880⟩ 259957

def event259968 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨2882⟩⟩) (.sum [.predecessor 0 259966 .coefficient, .predecessor 1 259967 .coefficient])

def event259969 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨2882⟩⟩) (.finite 655343)

def event259970 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5505⟩⟩) 0 ⟨2882⟩ 259969

def event259971 : Event := .predecessor (⟨.program ⟨257⟩, ⟨5505⟩⟩) 1 ⟨5426⟩ 259955

def event259972 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨5505⟩⟩) (.identity (.predecessor 1 259971 .coefficient))

def event259973 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨5505⟩⟩) (.finite 655360)

def event259974 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15354⟩⟩) 0 ⟨5505⟩ 259973

def event259975 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15354⟩⟩) (.authority (.programFamilyFact))

def exact259976RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15354⟩⟩], []⟩, (1)⟩]

theorem exact259976RawTermsValid :
    exact259976RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259976 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15354⟩⟩) exact259976RawTerms (.finite 2) 259975 .exactZero (none)

def event259977 : Event := .predecessor (⟨.program ⟨257⟩, ⟨12306⟩⟩) 0 ⟨5505⟩ 259973

def event259978 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨12306⟩⟩) (.authority (.programFamilyFact))

def exact259979RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12306⟩⟩], []⟩, (1)⟩]

theorem exact259979RawTermsValid :
    exact259979RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259979 : Event := .resultExact (⟨.program ⟨257⟩, ⟨12306⟩⟩) exact259979RawTerms (.finite 2) 259978 .exactZero (none)

def event259980 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15355⟩⟩) 0 ⟨12306⟩ 259979

def event259981 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15355⟩⟩) 1 ⟨15354⟩ 259976

def event259982 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15355⟩⟩) (.product (.predecessor 0 259980 .coefficient) (.predecessor 1 259981 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event259983 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15355⟩⟩, .operator (⟨259979, 0⟩, ⟨259976, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], []⟩, (1)⟩)

def exact259984RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], []⟩, (1)⟩]

theorem exact259984RawTermsValid :
    exact259984RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259984 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15355⟩⟩) exact259984RawTerms (.finite 4) 259982 .exactZero (none)

def event259985 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15356⟩⟩) 0 ⟨15355⟩ 259984

def event259986 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15356⟩⟩) (.identity (.predecessor 0 259985 .coefficient))

def event259987 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15356⟩⟩) (.finite 4)

def event259988 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15748⟩⟩) 0 ⟨15356⟩ 259987

def event259989 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15748⟩⟩) (.authority (.programFamilyFact))

def exact259990RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], []⟩, (1)⟩]

theorem exact259990RawTermsValid :
    exact259990RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event259990 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15748⟩⟩) exact259990RawTerms (.finite 2) 259989 .exactZero (none)

def event259991 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15749⟩⟩) 0 ⟨15748⟩ 259990

def event259992 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15749⟩⟩) (.identity (.predecessor 0 259991 .coefficient))

def event259993 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨15749⟩⟩) (.finite 2)

def event259994 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16954⟩⟩) 0 ⟨15749⟩ 259993

def event259995 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16954⟩⟩) (.authority (.programFamilyFact))

def event259996 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨16954⟩⟩) (.finite 3720)

def event259997 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨7177⟩⟩) .missing

def event259998 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16956⟩⟩) 0 ⟨7177⟩ 259997

def event259999 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16956⟩⟩) 1 ⟨16954⟩ 259996

def event260000 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16956⟩⟩) (.authority (.operator))

def exact260001RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨16956⟩⟩]⟩, (1)⟩]

theorem exact260001RawTermsValid :
    exact260001RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event260001 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16956⟩⟩) exact260001RawTerms .large 260000 .exactZero (none)

def event260002 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17621⟩⟩) 0 ⟨16956⟩ 260001

def event260003 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17621⟩⟩) (.authority (.operator))

def exact260004RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩, (1)⟩]

theorem exact260004RawTermsValid :
    exact260004RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event260004 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17621⟩⟩) exact260004RawTerms (.finite 8192) 260003 .exactZero (none)

def event260005 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨136⟩⟩) (.authority (.operator))

def event260006 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨136⟩⟩) .exactZero

def event260007 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17186⟩⟩) 0 ⟨15749⟩ 259993

def event260008 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17186⟩⟩) 1 ⟨136⟩ 260006

def event260009 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17186⟩⟩) (.sum [.predecessor 0 260007 .coefficient, .predecessor 1 260008 .coefficient])

def event260010 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨17186⟩⟩) (.finite 2)

def event260011 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17187⟩⟩) 0 ⟨17186⟩ 260010

def event260012 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17187⟩⟩) (.identity (.predecessor 0 260011 .coefficient))

def exact260013RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], []⟩, (1)⟩]

theorem exact260013RawTermsValid :
    exact260013RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event260013 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17187⟩⟩) exact260013RawTerms (.finite 2) 260012 .exactZero (none)

def event260014 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6908⟩⟩) (.authority (.factStore))

def exact260015RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact260015RawTermsValid :
    exact260015RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event260015 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6908⟩⟩) exact260015RawTerms .large 260014 .exactZero (none)

def event260016 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17188⟩⟩) 0 ⟨6908⟩ 260015

def event260017 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17188⟩⟩) 1 ⟨17187⟩ 260013

def event260018 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17188⟩⟩) (.product (.predecessor 0 260016 .coefficient) (.predecessor 1 260017 .coefficient) (⟨false, false, none, none, none⟩))

def event260019 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17188⟩⟩, .operator (⟨260015, 0⟩, ⟨260013, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact260020RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact260020RawTermsValid :
    exact260020RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event260020 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17188⟩⟩) exact260020RawTerms .large 260018 .exactZero (none)

def event260021 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7179⟩⟩) 0 ⟨7177⟩ 259997

def event260022 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7179⟩⟩) (.authority (.operator))

def exact260023RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩]

theorem exact260023RawTermsValid :
    exact260023RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event260023 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7179⟩⟩) exact260023RawTerms .large 260022 .exactZero (none)

def event260024 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17189⟩⟩) 0 ⟨7179⟩ 260023

def event260025 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17189⟩⟩) 1 ⟨17188⟩ 260020

def event260026 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17189⟩⟩) (.sum [.predecessor 0 260024 .coefficient, .predecessor 1 260025 .coefficient])

def exact260027RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact260027RawTermsValid :
    exact260027RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event260027 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17189⟩⟩) exact260027RawTerms .large 260026 .exactZero (none)

def event260028 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17622⟩⟩) 0 ⟨17189⟩ 260027

def event260029 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17622⟩⟩) 1 ⟨17621⟩ 260004

def event260030 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17622⟩⟩) (.product (.predecessor 0 260028 .coefficient) (.predecessor 1 260029 .coefficient) (⟨false, false, none, none, none⟩))

def event260031 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17622⟩⟩, .operator (⟨260027, 0⟩, ⟨260004, 0⟩), ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩, (1)⟩)

def event260032 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17622⟩⟩, .operator (⟨260027, 1⟩, ⟨260004, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩, (-1)⟩)

def event260033 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨17622⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩) (-1) 0 2 (.gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17621⟩⟩) ⟨16956⟩ 260001)

def event260034 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17622⟩⟩, .relation 260033 0, ⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨16956⟩⟩]⟩, (-1)⟩)

def exact260035RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨16956⟩⟩]⟩, (-1)⟩]

theorem exact260035RawTermsValid :
    exact260035RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event260035 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17622⟩⟩) exact260035RawTerms .large 260030 .exactZero (none)

def event260036 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15955⟩⟩) 0 ⟨15749⟩ 259993

def event260037 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15955⟩⟩) (.authority (.programFamilyFact))

def exact260038RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], []⟩, (1)⟩]

theorem exact260038RawTermsValid :
    exact260038RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event260038 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15955⟩⟩) exact260038RawTerms (.finite 43) 260037 .exactZero (none)

def event260039 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15956⟩⟩) 0 ⟨6908⟩ 260015

def event260040 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15956⟩⟩) 1 ⟨15955⟩ 260038

def event260041 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15956⟩⟩) (.product (.predecessor 0 260039 .coefficient) (.predecessor 1 260040 .coefficient) (⟨false, true, none, none, some 1⟩))

def event260042 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨15956⟩⟩, .operator (⟨260015, 0⟩, ⟨260038, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩)

def exact260043RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (1)⟩]

theorem exact260043RawTermsValid :
    exact260043RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event260043 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15956⟩⟩) exact260043RawTerms .large 260041 .exactZero (none)

def event260044 : Event := .predecessor (⟨.program ⟨257⟩, ⟨7198⟩⟩) 0 ⟨7177⟩ 259997

def event260045 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨7198⟩⟩) (.authority (.operator))

def exact260046RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩]

theorem exact260046RawTermsValid :
    exact260046RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event260046 : Event := .resultExact (⟨.program ⟨257⟩, ⟨7198⟩⟩) exact260046RawTerms .large 260045 .exactZero (none)

def event260047 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15957⟩⟩) 0 ⟨7198⟩ 260046

def event260048 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15957⟩⟩) 1 ⟨15956⟩ 260043

def event260049 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15957⟩⟩) (.sum [.predecessor 0 260047 .coefficient, .predecessor 1 260048 .coefficient])

def exact260050RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact260050RawTermsValid :
    exact260050RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event260050 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15957⟩⟩) exact260050RawTerms .large 260049 .exactZero (none)

def event260051 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17625⟩⟩) 0 ⟨15957⟩ 260050

def event260052 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17625⟩⟩) 1 ⟨17622⟩ 260035

def event260053 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17625⟩⟩) (.sum [.predecessor 0 260051 .coefficient, .predecessor 1 260052 .coefficient])

def exact260054RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨16956⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact260054RawTermsValid :
    exact260054RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event260054 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17625⟩⟩) exact260054RawTerms .large 260053 .exactZero (none)

def event260055 : Event := .preFoldPolynomial 260054 [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨16956⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩] .exactZero none

def exact260056RawTerms : List Term := [⟨⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩, (-1)⟩, ⟨⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨16956⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

def event260056 : Event := .invocationEndExact (⟨.program ⟨257⟩, ⟨17625⟩⟩) 260055 exact260056RawTerms .large 260053 .exactZero (none)

def event260057 : Event := .specializationComputed (⟨.program ⟨257⟩, ⟨15749⟩⟩) ⟨⟨77⟩, ⟨57⟩, ⟨135⟩⟩ ⟨259899, 260057⟩

def event260058 : Event := .appliedRelation (⟨.program ⟨257⟩, ⟨16499⟩⟩) (⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16496⟩⟩]⟩) (1) 0 2 (.universal 260057 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16496⟩⟩]⟩) (none) 260056)

def event260059 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16499⟩⟩, .relation 260058 1, ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩)

def event260060 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16499⟩⟩, .relation 260058 0, ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩, (-1)⟩)

def event260061 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16499⟩⟩, .relation 260058 2, ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨16956⟩⟩]⟩, (1)⟩)

def event260062 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16499⟩⟩, .relation 260058 3, ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩)

def exact260063RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨16956⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact260063RawTermsValid :
    exact260063RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event260063 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16499⟩⟩) exact260063RawTerms .large 259895 (.finite 202072841853861888) (some (259897))

def event260064 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17624⟩⟩) 0 ⟨16499⟩ 260063

def event260065 : Event := .predecessor (⟨.program ⟨257⟩, ⟨17624⟩⟩) 1 ⟨17623⟩ 259885

def event260066 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17624⟩⟩) (.sum [.predecessor 0 260064 .coefficient, .predecessor 1 260065 .coefficient])

def event260067 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17624⟩⟩, .operator (⟨260063, 0⟩, ⟨259885, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩, (1)⟩)

def event260068 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨17624⟩⟩, .operator (⟨260063, 2⟩, ⟨259885, 1⟩), ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨16956⟩⟩]⟩, (-1)⟩)

def event260069 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨17624⟩⟩) (.sum [.result 260063 .summary, .result 259885 .summary])

def exact260070RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact260070RawTermsValid :
    exact260070RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event260070 : Event := .resultExact (⟨.program ⟨257⟩, ⟨17624⟩⟩) exact260070RawTerms .large 260066 (.finite 32188807212483706889510625476608) (some (260069))

def event260071 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20501⟩⟩) 0 ⟨17624⟩ 260070

def event260072 : Event := .predecessor (⟨.program ⟨257⟩, ⟨20501⟩⟩) 1 ⟨20500⟩ 259588

def event260073 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20501⟩⟩) (.sum [.predecessor 0 260071 .coefficient, .predecessor 1 260072 .coefficient])

def event260074 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨20501⟩⟩) (.sum [.result 260070 .summary, .result 259588 .summary])

def exact260075RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨18771⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact260075RawTermsValid :
    exact260075RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event260075 : Event := .resultExact (⟨.program ⟨257⟩, ⟨20501⟩⟩) exact260075RawTerms .large 260073 (.finite 64377712650190257467641695830016) (some (260074))

def event260076 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23721⟩⟩) 0 ⟨20501⟩ 260075

def event260077 : Event := .predecessor (⟨.program ⟨257⟩, ⟨23721⟩⟩) 1 ⟨23720⟩ 259106

def event260078 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23721⟩⟩) (.sum [.predecessor 0 260076 .coefficient, .predecessor 1 260077 .coefficient])

def event260079 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨23721⟩⟩) (.sum [.result 260075 .summary, .result 259106 .summary])

def exact260080RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨18771⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨21991⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact260080RawTermsValid :
    exact260080RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event260080 : Event := .resultExact (⟨.program ⟨257⟩, ⟨23721⟩⟩) exact260080RawTerms .large 260078 (.finite 96566716313119651734393211060224) (some (260079))

def event260081 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33741⟩⟩) 0 ⟨23721⟩ 260080

def event260082 : Event := .predecessor (⟨.program ⟨257⟩, ⟨33741⟩⟩) 1 ⟨33740⟩ 258624

def event260083 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33741⟩⟩) (.sum [.predecessor 0 260081 .coefficient, .predecessor 1 260082 .coefficient])

def event260084 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨33741⟩⟩) (.sum [.result 260080 .summary, .result 258624 .summary])

def exact260085RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨18771⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨21991⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨32011⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact260085RawTermsValid :
    exact260085RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event260085 : Event := .resultExact (⟨.program ⟨257⟩, ⟨33741⟩⟩) exact260085RawTerms .large 260083 (.finite 128755916426494733378385616044032) (some (260084))

def event260086 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52801⟩⟩) 0 ⟨33741⟩ 260085

def event260087 : Event := .predecessor (⟨.program ⟨257⟩, ⟨52801⟩⟩) 1 ⟨52800⟩ 258142

def event260088 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52801⟩⟩) (.sum [.predecessor 0 260086 .coefficient, .predecessor 1 260087 .coefficient])

def event260089 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨52801⟩⟩) (.sum [.result 260085 .summary, .result 258142 .summary])

def exact260090RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨18771⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨21991⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨32011⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨51066⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact260090RawTermsValid :
    exact260090RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event260090 : Event := .resultExact (⟨.program ⟨257⟩, ⟨52801⟩⟩) exact260090RawTerms .large 260088 (.finite 160945509440761189776859800535040) (some (260089))

def event260091 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55781⟩⟩) 0 ⟨52801⟩ 260090

def event260092 : Event := .predecessor (⟨.program ⟨257⟩, ⟨55781⟩⟩) 1 ⟨55780⟩ 257660

def event260093 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55781⟩⟩) (.sum [.predecessor 0 260091 .coefficient, .predecessor 1 260092 .coefficient])

def event260094 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨55781⟩⟩) (.sum [.result 260090 .summary, .result 257660 .summary])

def exact260095RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨18771⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨21991⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨32011⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨51066⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨54046⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩, (-1)⟩]

theorem exact260095RawTermsValid :
    exact260095RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event260095 : Event := .resultExact (⟨.program ⟨257⟩, ⟨55781⟩⟩) exact260095RawTerms .large 260093 (.finite 193135298905473333552574874779648) (some (260094))

def eventLeaf16240 : Array AnnotatedEvent := #[
  { event := event259840
    frameStart := 259744 },
  { event := event259841
    frameStart := 259744 },
  { event := event259842
    frameStart := 259744 },
  { event := event259843
    frameStart := 259744 },
  { event := event259844
    frameStart := 259744 },
  { event := event259845
    frameStart := 259744 },
  { event := event259846
    frameStart := 259744 },
  { event := event259847
    frameStart := 259744 },
  { event := event259848
    frameStart := 259744 },
  { event := event259849
    frameStart := 259744 },
  { event := event259850
    frameStart := 259744 },
  { event := event259851
    frameStart := 259744 },
  { event := event259852
    frameStart := 259744 },
  { event := event259853
    frameStart := 259744 },
  { event := event259854
    frameStart := 259744 },
  { event := event259855
    frameStart := 259744 }
]

def eventLeaf16241 : Array AnnotatedEvent := #[
  { event := event259856
    frameStart := 259744 },
  { event := event259857
    frameStart := 259744 },
  { event := event259858
    frameStart := 259744 },
  { event := event259859
    frameStart := 259744 },
  { event := event259860
    frameStart := 259744 },
  { event := event259861
    frameStart := 259744 },
  { event := event259862
    frameStart := 0 },
  { event := event259863
    frameStart := 0 },
  { event := event259864
    frameStart := 0 },
  { event := event259865
    frameStart := 0 },
  { event := event259866
    frameStart := 0 },
  { event := event259867
    frameStart := 0 },
  { event := event259868
    frameStart := 0 },
  { event := event259869
    frameStart := 0 },
  { event := event259870
    frameStart := 0 },
  { event := event259871
    frameStart := 0 }
]

def eventLeaf16242 : Array AnnotatedEvent := #[
  { event := event259872
    frameStart := 0 },
  { event := event259873
    frameStart := 0 },
  { event := event259874
    frameStart := 0 },
  { event := event259875
    frameStart := 0 },
  { event := event259876
    frameStart := 0 },
  { event := event259877
    frameStart := 0 },
  { event := event259878
    frameStart := 0 },
  { event := event259879
    frameStart := 0 },
  { event := event259880
    frameStart := 0 },
  { event := event259881
    frameStart := 0 },
  { event := event259882
    frameStart := 0 },
  { event := event259883
    frameStart := 0 },
  { event := event259884
    frameStart := 0 },
  { event := event259885
    frameStart := 0 },
  { event := event259886
    frameStart := 0 },
  { event := event259887
    frameStart := 0 }
]

def eventLeaf16243 : Array AnnotatedEvent := #[
  { event := event259888
    frameStart := 0 },
  { event := event259889
    frameStart := 0 },
  { event := event259890
    frameStart := 0 },
  { event := event259891
    frameStart := 0 },
  { event := event259892
    frameStart := 0 },
  { event := event259893
    frameStart := 0 },
  { event := event259894
    frameStart := 0 },
  { event := event259895
    frameStart := 0 },
  { event := event259896
    frameStart := 0 },
  { event := event259897
    frameStart := 0 },
  { event := event259898
    frameStart := 0 },
  { event := event259899
    frameStart := 259899 },
  { event := event259900
    frameStart := 259899 },
  { event := event259901
    frameStart := 259899 },
  { event := event259902
    frameStart := 259899 },
  { event := event259903
    frameStart := 259899 }
]

def eventLeaf16244 : Array AnnotatedEvent := #[
  { event := event259904
    frameStart := 259899 },
  { event := event259905
    frameStart := 259899 },
  { event := event259906
    frameStart := 259899 },
  { event := event259907
    frameStart := 259899 },
  { event := event259908
    frameStart := 259899 },
  { event := event259909
    frameStart := 259899 },
  { event := event259910
    frameStart := 259899 },
  { event := event259911
    frameStart := 259899 },
  { event := event259912
    frameStart := 259899 },
  { event := event259913
    frameStart := 259899 },
  { event := event259914
    frameStart := 259899 },
  { event := event259915
    frameStart := 259899 },
  { event := event259916
    frameStart := 259899 },
  { event := event259917
    frameStart := 259899 },
  { event := event259918
    frameStart := 259899 },
  { event := event259919
    frameStart := 259899 }
]

def eventLeaf16245 : Array AnnotatedEvent := #[
  { event := event259920
    frameStart := 259899 },
  { event := event259921
    frameStart := 259899 },
  { event := event259922
    frameStart := 259899 },
  { event := event259923
    frameStart := 259899 },
  { event := event259924
    frameStart := 259899 },
  { event := event259925
    frameStart := 259899 },
  { event := event259926
    frameStart := 259899 },
  { event := event259927
    frameStart := 259899 },
  { event := event259928
    frameStart := 259899 },
  { event := event259929
    frameStart := 259899 },
  { event := event259930
    frameStart := 259899 },
  { event := event259931
    frameStart := 259899 },
  { event := event259932
    frameStart := 259899 },
  { event := event259933
    frameStart := 259899 },
  { event := event259934
    frameStart := 259899 },
  { event := event259935
    frameStart := 259899 }
]

def eventLeaf16246 : Array AnnotatedEvent := #[
  { event := event259936
    frameStart := 259899 },
  { event := event259937
    frameStart := 259899 },
  { event := event259938
    frameStart := 259899 },
  { event := event259939
    frameStart := 259899 },
  { event := event259940
    frameStart := 259899 },
  { event := event259941
    frameStart := 259899 },
  { event := event259942
    frameStart := 259899 },
  { event := event259943
    frameStart := 259899 },
  { event := event259944
    frameStart := 259899 },
  { event := event259945
    frameStart := 259899 },
  { event := event259946
    frameStart := 259899 },
  { event := event259947
    frameStart := 259899 },
  { event := event259948
    frameStart := 259899 },
  { event := event259949
    frameStart := 259899 },
  { event := event259950
    frameStart := 259899 },
  { event := event259951
    frameStart := 259899 }
]

def eventLeaf16247 : Array AnnotatedEvent := #[
  { event := event259952
    frameStart := 259899 },
  { event := event259953
    frameStart := 259953 },
  { event := event259954
    frameStart := 259953 },
  { event := event259955
    frameStart := 259953 },
  { event := event259956
    frameStart := 259953 },
  { event := event259957
    frameStart := 259953 },
  { event := event259958
    frameStart := 259953 },
  { event := event259959
    frameStart := 259953 },
  { event := event259960
    frameStart := 259953 },
  { event := event259961
    frameStart := 259953 },
  { event := event259962
    frameStart := 259953 },
  { event := event259963
    frameStart := 259953 },
  { event := event259964
    frameStart := 259953 },
  { event := event259965
    frameStart := 259953 },
  { event := event259966
    frameStart := 259953 },
  { event := event259967
    frameStart := 259953 }
]

def eventLeaf16248 : Array AnnotatedEvent := #[
  { event := event259968
    frameStart := 259953 },
  { event := event259969
    frameStart := 259953 },
  { event := event259970
    frameStart := 259953 },
  { event := event259971
    frameStart := 259953 },
  { event := event259972
    frameStart := 259953 },
  { event := event259973
    frameStart := 259953 },
  { event := event259974
    frameStart := 259953 },
  { event := event259975
    frameStart := 259953 },
  { event := event259976
    frameStart := 259953 },
  { event := event259977
    frameStart := 259953 },
  { event := event259978
    frameStart := 259953 },
  { event := event259979
    frameStart := 259953 },
  { event := event259980
    frameStart := 259953 },
  { event := event259981
    frameStart := 259953 },
  { event := event259982
    frameStart := 259953 },
  { event := event259983
    frameStart := 259953 }
]

def eventLeaf16249 : Array AnnotatedEvent := #[
  { event := event259984
    frameStart := 259953 },
  { event := event259985
    frameStart := 259953 },
  { event := event259986
    frameStart := 259953 },
  { event := event259987
    frameStart := 259953 },
  { event := event259988
    frameStart := 259953 },
  { event := event259989
    frameStart := 259953 },
  { event := event259990
    frameStart := 259953 },
  { event := event259991
    frameStart := 259953 },
  { event := event259992
    frameStart := 259953 },
  { event := event259993
    frameStart := 259953 },
  { event := event259994
    frameStart := 259953 },
  { event := event259995
    frameStart := 259953 },
  { event := event259996
    frameStart := 259953 },
  { event := event259997
    frameStart := 259953 },
  { event := event259998
    frameStart := 259953 },
  { event := event259999
    frameStart := 259953 }
]

def eventLeaf16250 : Array AnnotatedEvent := #[
  { event := event260000
    frameStart := 259953 },
  { event := event260001
    frameStart := 259953 },
  { event := event260002
    frameStart := 259953 },
  { event := event260003
    frameStart := 259953 },
  { event := event260004
    frameStart := 259953 },
  { event := event260005
    frameStart := 259953 },
  { event := event260006
    frameStart := 259953 },
  { event := event260007
    frameStart := 259953 },
  { event := event260008
    frameStart := 259953 },
  { event := event260009
    frameStart := 259953 },
  { event := event260010
    frameStart := 259953 },
  { event := event260011
    frameStart := 259953 },
  { event := event260012
    frameStart := 259953 },
  { event := event260013
    frameStart := 259953 },
  { event := event260014
    frameStart := 259953 },
  { event := event260015
    frameStart := 259953 }
]

def eventLeaf16251 : Array AnnotatedEvent := #[
  { event := event260016
    frameStart := 259953 },
  { event := event260017
    frameStart := 259953 },
  { event := event260018
    frameStart := 259953 },
  { event := event260019
    frameStart := 259953 },
  { event := event260020
    frameStart := 259953 },
  { event := event260021
    frameStart := 259953 },
  { event := event260022
    frameStart := 259953 },
  { event := event260023
    frameStart := 259953 },
  { event := event260024
    frameStart := 259953 },
  { event := event260025
    frameStart := 259953 },
  { event := event260026
    frameStart := 259953 },
  { event := event260027
    frameStart := 259953 },
  { event := event260028
    frameStart := 259953 },
  { event := event260029
    frameStart := 259953 },
  { event := event260030
    frameStart := 259953 },
  { event := event260031
    frameStart := 259953 }
]

def eventLeaf16252 : Array AnnotatedEvent := #[
  { event := event260032
    frameStart := 259953 },
  { event := event260033
    frameStart := 259953 },
  { event := event260034
    frameStart := 259953 },
  { event := event260035
    frameStart := 259953 },
  { event := event260036
    frameStart := 259953 },
  { event := event260037
    frameStart := 259953 },
  { event := event260038
    frameStart := 259953 },
  { event := event260039
    frameStart := 259953 },
  { event := event260040
    frameStart := 259953 },
  { event := event260041
    frameStart := 259953 },
  { event := event260042
    frameStart := 259953 },
  { event := event260043
    frameStart := 259953 },
  { event := event260044
    frameStart := 259953 },
  { event := event260045
    frameStart := 259953 },
  { event := event260046
    frameStart := 259953 },
  { event := event260047
    frameStart := 259953 }
]

def eventLeaf16253 : Array AnnotatedEvent := #[
  { event := event260048
    frameStart := 259953 },
  { event := event260049
    frameStart := 259953 },
  { event := event260050
    frameStart := 259953 },
  { event := event260051
    frameStart := 259953 },
  { event := event260052
    frameStart := 259953 },
  { event := event260053
    frameStart := 259953 },
  { event := event260054
    frameStart := 259953 },
  { event := event260055
    frameStart := 259953 },
  { event := event260056
    frameStart := 259953 },
  { event := event260057
    frameStart := 0 },
  { event := event260058
    frameStart := 0 },
  { event := event260059
    frameStart := 0 },
  { event := event260060
    frameStart := 0 },
  { event := event260061
    frameStart := 0 },
  { event := event260062
    frameStart := 0 },
  { event := event260063
    frameStart := 0 }
]

def eventLeaf16254 : Array AnnotatedEvent := #[
  { event := event260064
    frameStart := 0 },
  { event := event260065
    frameStart := 0 },
  { event := event260066
    frameStart := 0 },
  { event := event260067
    frameStart := 0 },
  { event := event260068
    frameStart := 0 },
  { event := event260069
    frameStart := 0 },
  { event := event260070
    frameStart := 0 },
  { event := event260071
    frameStart := 0 },
  { event := event260072
    frameStart := 0 },
  { event := event260073
    frameStart := 0 },
  { event := event260074
    frameStart := 0 },
  { event := event260075
    frameStart := 0 },
  { event := event260076
    frameStart := 0 },
  { event := event260077
    frameStart := 0 },
  { event := event260078
    frameStart := 0 },
  { event := event260079
    frameStart := 0 }
]

def eventLeaf16255 : Array AnnotatedEvent := #[
  { event := event260080
    frameStart := 0 },
  { event := event260081
    frameStart := 0 },
  { event := event260082
    frameStart := 0 },
  { event := event260083
    frameStart := 0 },
  { event := event260084
    frameStart := 0 },
  { event := event260085
    frameStart := 0 },
  { event := event260086
    frameStart := 0 },
  { event := event260087
    frameStart := 0 },
  { event := event260088
    frameStart := 0 },
  { event := event260089
    frameStart := 0 },
  { event := event260090
    frameStart := 0 },
  { event := event260091
    frameStart := 0 },
  { event := event260092
    frameStart := 0 },
  { event := event260093
    frameStart := 0 },
  { event := event260094
    frameStart := 0 },
  { event := event260095
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events1015
