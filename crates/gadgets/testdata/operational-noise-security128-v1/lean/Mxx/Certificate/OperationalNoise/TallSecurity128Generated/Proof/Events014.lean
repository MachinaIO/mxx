import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.Cert

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events014

open Mxx.Certificate.OperationalNoise
open CertificateABI

def exact3584RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], []⟩, (1)⟩]

theorem exact3584RawTermsValid :
    exact3584RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3584 : Event := .resultExact (⟨.program ⟨257⟩, ⟨48438⟩⟩) exact3584RawTerms (.finite 230731242018505516688400) 3582 .exactZero (none)

def event3585 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45757⟩⟩) 0 ⟨45517⟩ 3126

def event3586 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45757⟩⟩) (.authority (.programFamilyFact))

def exact3587RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨45757⟩⟩], []⟩, (1)⟩]

theorem exact3587RawTermsValid :
    exact3587RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3587 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45757⟩⟩) exact3587RawTerms (.finite 58) 3586 .exactZero (none)

def event3588 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45758⟩⟩) 0 ⟨45757⟩ 3587

def event3589 : Event := .predecessor (⟨.program ⟨257⟩, ⟨45758⟩⟩) 1 ⟨6807⟩ 553

def event3590 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨45758⟩⟩) (.product (.predecessor 0 3588 .coefficient) (.predecessor 1 3589 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3591 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨45758⟩⟩, .operator (⟨3587, 0⟩, ⟨553, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], []⟩, (1)⟩)

def exact3592RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], []⟩, (1)⟩]

theorem exact3592RawTermsValid :
    exact3592RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3592 : Event := .resultExact (⟨.program ⟨257⟩, ⟨45758⟩⟩) exact3592RawTerms (.finite 230600885384596756509480) 3590 .exactZero (none)

def event3593 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43080⟩⟩) 0 ⟨42837⟩ 3149

def event3594 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43080⟩⟩) (.authority (.programFamilyFact))

def exact3595RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨43080⟩⟩], []⟩, (1)⟩]

theorem exact3595RawTermsValid :
    exact3595RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3595 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43080⟩⟩) exact3595RawTerms (.finite 52) 3594 .exactZero (none)

def event3596 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43081⟩⟩) 0 ⟨43080⟩ 3595

def event3597 : Event := .predecessor (⟨.program ⟨257⟩, ⟨43081⟩⟩) 1 ⟨6817⟩ 563

def event3598 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨43081⟩⟩) (.product (.predecessor 0 3596 .coefficient) (.predecessor 1 3597 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3599 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨43081⟩⟩, .operator (⟨3595, 0⟩, ⟨563, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], []⟩, (1)⟩)

def exact3600RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], []⟩, (1)⟩]

theorem exact3600RawTermsValid :
    exact3600RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3600 : Event := .resultExact (⟨.program ⟨257⟩, ⟨43081⟩⟩) exact3600RawTerms (.finite 230150786063741980797360) 3598 .exactZero (none)

def event3601 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40400⟩⟩) 0 ⟨40157⟩ 3172

def event3602 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40400⟩⟩) (.authority (.programFamilyFact))

def exact3603RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨40400⟩⟩], []⟩, (1)⟩]

theorem exact3603RawTermsValid :
    exact3603RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3603 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40400⟩⟩) exact3603RawTerms (.finite 46) 3602 .exactZero (none)

def event3604 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40401⟩⟩) 0 ⟨40400⟩ 3603

def event3605 : Event := .predecessor (⟨.program ⟨257⟩, ⟨40401⟩⟩) 1 ⟨6828⟩ 573

def event3606 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨40401⟩⟩) (.product (.predecessor 0 3604 .coefficient) (.predecessor 1 3605 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3607 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨40401⟩⟩, .operator (⟨3603, 0⟩, ⟨573, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], []⟩, (1)⟩)

def exact3608RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], []⟩, (1)⟩]

theorem exact3608RawTermsValid :
    exact3608RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3608 : Event := .resultExact (⟨.program ⟨257⟩, ⟨40401⟩⟩) exact3608RawTerms (.finite 229585767767349815541720) 3606 .exactZero (none)

def event3609 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37717⟩⟩) 0 ⟨37477⟩ 3195

def event3610 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37717⟩⟩) (.authority (.programFamilyFact))

def exact3611RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨37717⟩⟩], []⟩, (1)⟩]

theorem exact3611RawTermsValid :
    exact3611RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3611 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37717⟩⟩) exact3611RawTerms (.finite 42) 3610 .exactZero (none)

def event3612 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37718⟩⟩) 0 ⟨37717⟩ 3611

def event3613 : Event := .predecessor (⟨.program ⟨257⟩, ⟨37718⟩⟩) 1 ⟨6838⟩ 583

def event3614 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨37718⟩⟩) (.product (.predecessor 0 3612 .coefficient) (.predecessor 1 3613 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3615 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨37718⟩⟩, .operator (⟨3611, 0⟩, ⟨583, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], []⟩, (1)⟩)

def exact3616RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], []⟩, (1)⟩]

theorem exact3616RawTermsValid :
    exact3616RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3616 : Event := .resultExact (⟨.program ⟨257⟩, ⟨37718⟩⟩) exact3616RawTerms (.finite 229121489167213617734760) 3614 .exactZero (none)

def event3617 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35037⟩⟩) 0 ⟨34797⟩ 3218

def event3618 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35037⟩⟩) (.authority (.programFamilyFact))

def exact3619RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨35037⟩⟩], []⟩, (1)⟩]

theorem exact3619RawTermsValid :
    exact3619RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3619 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35037⟩⟩) exact3619RawTerms (.finite 40) 3618 .exactZero (none)

def event3620 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35038⟩⟩) 0 ⟨35037⟩ 3619

def event3621 : Event := .predecessor (⟨.program ⟨257⟩, ⟨35038⟩⟩) 1 ⟨6842⟩ 593

def event3622 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨35038⟩⟩) (.product (.predecessor 0 3620 .coefficient) (.predecessor 1 3621 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3623 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨35038⟩⟩, .operator (⟨3619, 0⟩, ⟨593, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], []⟩, (1)⟩)

def exact3624RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], []⟩, (1)⟩]

theorem exact3624RawTermsValid :
    exact3624RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3624 : Event := .resultExact (⟨.program ⟨257⟩, ⟨35038⟩⟩) exact3624RawTerms (.finite 228855378262257504357600) 3622 .exactZero (none)

def event3625 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29380⟩⟩) 0 ⟨29137⟩ 3241

def event3626 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29380⟩⟩) (.authority (.programFamilyFact))

def exact3627RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨29380⟩⟩], []⟩, (1)⟩]

theorem exact3627RawTermsValid :
    exact3627RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3627 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29380⟩⟩) exact3627RawTerms (.finite 36) 3626 .exactZero (none)

def event3628 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29381⟩⟩) 0 ⟨29380⟩ 3627

def event3629 : Event := .predecessor (⟨.program ⟨257⟩, ⟨29381⟩⟩) 1 ⟨6857⟩ 603

def event3630 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨29381⟩⟩) (.product (.predecessor 0 3628 .coefficient) (.predecessor 1 3629 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3631 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨29381⟩⟩, .operator (⟨3627, 0⟩, ⟨603, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], []⟩, (1)⟩)

def exact3632RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], []⟩, (1)⟩]

theorem exact3632RawTermsValid :
    exact3632RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3632 : Event := .resultExact (⟨.program ⟨257⟩, ⟨29381⟩⟩) exact3632RawTerms (.finite 228236850212900051643120) 3630 .exactZero (none)

def event3633 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26700⟩⟩) 0 ⟨26457⟩ 3264

def event3634 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26700⟩⟩) (.authority (.programFamilyFact))

def exact3635RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨26700⟩⟩], []⟩, (1)⟩]

theorem exact3635RawTermsValid :
    exact3635RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3635 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26700⟩⟩) exact3635RawTerms (.finite 30) 3634 .exactZero (none)

def event3636 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26701⟩⟩) 0 ⟨26700⟩ 3635

def event3637 : Event := .predecessor (⟨.program ⟨257⟩, ⟨26701⟩⟩) 1 ⟨6860⟩ 613

def event3638 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨26701⟩⟩) (.product (.predecessor 0 3636 .coefficient) (.predecessor 1 3637 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3639 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨26701⟩⟩, .operator (⟨3635, 0⟩, ⟨613, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], []⟩, (1)⟩)

def exact3640RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], []⟩, (1)⟩]

theorem exact3640RawTermsValid :
    exact3640RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3640 : Event := .resultExact (⟨.program ⟨257⟩, ⟨26701⟩⟩) exact3640RawTerms (.finite 227009770373045750290200) 3638 .exactZero (none)

def event3641 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67008⟩⟩) 0 ⟨65837⟩ 3287

def event3642 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67008⟩⟩) (.authority (.programFamilyFact))

def exact3643RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨67008⟩⟩], []⟩, (1)⟩]

theorem exact3643RawTermsValid :
    exact3643RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3643 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67008⟩⟩) exact3643RawTerms (.finite 28) 3642 .exactZero (none)

def event3644 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67009⟩⟩) 0 ⟨67008⟩ 3643

def event3645 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67009⟩⟩) 1 ⟨6870⟩ 623

def event3646 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67009⟩⟩) (.product (.predecessor 0 3644 .coefficient) (.predecessor 1 3645 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3647 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67009⟩⟩, .operator (⟨3643, 0⟩, ⟨623, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], []⟩, (1)⟩)

def exact3648RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], []⟩, (1)⟩]

theorem exact3648RawTermsValid :
    exact3648RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3648 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67009⟩⟩) exact3648RawTerms (.finite 226487908831958288795280) 3646 .exactZero (none)

def event3649 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63199⟩⟩) 0 ⟨62857⟩ 3310

def event3650 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63199⟩⟩) (.authority (.programFamilyFact))

def exact3651RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨63199⟩⟩], []⟩, (1)⟩]

theorem exact3651RawTermsValid :
    exact3651RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3651 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63199⟩⟩) exact3651RawTerms (.finite 22) 3650 .exactZero (none)

def event3652 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63200⟩⟩) 0 ⟨63199⟩ 3651

def event3653 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63200⟩⟩) 1 ⟨6732⟩ 633

def event3654 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63200⟩⟩) (.product (.predecessor 0 3652 .coefficient) (.predecessor 1 3653 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3655 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨63200⟩⟩, .operator (⟨3651, 0⟩, ⟨633, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], []⟩, (1)⟩)

def exact3656RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], []⟩, (1)⟩]

theorem exact3656RawTermsValid :
    exact3656RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3656 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63200⟩⟩) exact3656RawTerms (.finite 224377773035387248837560) 3654 .exactZero (none)

def event3657 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60219⟩⟩) 0 ⟨59877⟩ 3333

def event3658 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60219⟩⟩) (.authority (.programFamilyFact))

def exact3659RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨60219⟩⟩], []⟩, (1)⟩]

theorem exact3659RawTermsValid :
    exact3659RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3659 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60219⟩⟩) exact3659RawTerms (.finite 18) 3658 .exactZero (none)

def event3660 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60220⟩⟩) 0 ⟨60219⟩ 3659

def event3661 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60220⟩⟩) 1 ⟨6736⟩ 643

def event3662 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60220⟩⟩) (.product (.predecessor 0 3660 .coefficient) (.predecessor 1 3661 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3663 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨60220⟩⟩, .operator (⟨3659, 0⟩, ⟨643, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], []⟩, (1)⟩)

def exact3664RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], []⟩, (1)⟩]

theorem exact3664RawTermsValid :
    exact3664RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3664 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60220⟩⟩) exact3664RawTerms (.finite 222230617312560576599880) 3662 .exactZero (none)

def event3665 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57239⟩⟩) 0 ⟨56897⟩ 3356

def event3666 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57239⟩⟩) (.authority (.programFamilyFact))

def exact3667RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨57239⟩⟩], []⟩, (1)⟩]

theorem exact3667RawTermsValid :
    exact3667RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3667 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57239⟩⟩) exact3667RawTerms (.finite 16) 3666 .exactZero (none)

def event3668 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57240⟩⟩) 0 ⟨57239⟩ 3667

def event3669 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57240⟩⟩) 1 ⟨6741⟩ 653

def event3670 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57240⟩⟩) (.product (.predecessor 0 3668 .coefficient) (.predecessor 1 3669 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3671 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨57240⟩⟩, .operator (⟨3667, 0⟩, ⟨653, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], []⟩, (1)⟩)

def exact3672RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], []⟩, (1)⟩]

theorem exact3672RawTermsValid :
    exact3672RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3672 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57240⟩⟩) exact3672RawTerms (.finite 220778129617707239497920) 3670 .exactZero (none)

def event3673 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54259⟩⟩) 0 ⟨53917⟩ 3379

def event3674 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54259⟩⟩) (.authority (.programFamilyFact))

def exact3675RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨54259⟩⟩], []⟩, (1)⟩]

theorem exact3675RawTermsValid :
    exact3675RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3675 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54259⟩⟩) exact3675RawTerms (.finite 12) 3674 .exactZero (none)

def event3676 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54260⟩⟩) 0 ⟨54259⟩ 3675

def event3677 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54260⟩⟩) 1 ⟨6757⟩ 663

def event3678 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54260⟩⟩) (.product (.predecessor 0 3676 .coefficient) (.predecessor 1 3677 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3679 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨54260⟩⟩, .operator (⟨3675, 0⟩, ⟨663, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], []⟩, (1)⟩)

def exact3680RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], []⟩, (1)⟩]

theorem exact3680RawTermsValid :
    exact3680RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3680 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54260⟩⟩) exact3680RawTerms (.finite 216532396355828254122960) 3678 .exactZero (none)

def event3681 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51279⟩⟩) 0 ⟨50937⟩ 3402

def event3682 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51279⟩⟩) (.authority (.programFamilyFact))

def exact3683RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩, (1)⟩]

theorem exact3683RawTermsValid :
    exact3683RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3683 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51279⟩⟩) exact3683RawTerms (.finite 10) 3682 .exactZero (none)

def event3684 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51280⟩⟩) 0 ⟨51279⟩ 3683

def event3685 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51280⟩⟩) 1 ⟨6768⟩ 673

def event3686 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51280⟩⟩) (.product (.predecessor 0 3684 .coefficient) (.predecessor 1 3685 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3687 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨51280⟩⟩, .operator (⟨3683, 0⟩, ⟨673, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩, (1)⟩)

def exact3688RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩, (1)⟩]

theorem exact3688RawTermsValid :
    exact3688RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3688 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51280⟩⟩) exact3688RawTerms (.finite 213251602471649038151400) 3686 .exactZero (none)

def event3689 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32215⟩⟩) 0 ⟨31877⟩ 3425

def event3690 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32215⟩⟩) (.authority (.programFamilyFact))

def exact3691RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩, (1)⟩]

theorem exact3691RawTermsValid :
    exact3691RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3691 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32215⟩⟩) exact3691RawTerms (.finite 6) 3690 .exactZero (none)

def event3692 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32216⟩⟩) 0 ⟨32215⟩ 3691

def event3693 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32216⟩⟩) 1 ⟨6794⟩ 683

def event3694 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32216⟩⟩) (.product (.predecessor 0 3692 .coefficient) (.predecessor 1 3693 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3695 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨32216⟩⟩, .operator (⟨3691, 0⟩, ⟨683, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩, (1)⟩)

def exact3696RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩, (1)⟩]

theorem exact3696RawTermsValid :
    exact3696RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3696 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32216⟩⟩) exact3696RawTerms (.finite 201065796616126235971320) 3694 .exactZero (none)

def event3697 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22195⟩⟩) 0 ⟨21857⟩ 3448

def event3698 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22195⟩⟩) (.authority (.programFamilyFact))

def exact3699RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩]

theorem exact3699RawTermsValid :
    exact3699RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3699 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22195⟩⟩) exact3699RawTerms (.finite 4) 3698 .exactZero (none)

def event3700 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22196⟩⟩) 0 ⟨22195⟩ 3699

def event3701 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22196⟩⟩) 1 ⟨6822⟩ 693

def event3702 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22196⟩⟩) (.product (.predecessor 0 3700 .coefficient) (.predecessor 1 3701 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3703 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨22196⟩⟩, .operator (⟨3699, 0⟩, ⟨693, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩)

def exact3704RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩]

theorem exact3704RawTermsValid :
    exact3704RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3704 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22196⟩⟩) exact3704RawTerms (.finite 187661410175051153573232) 3702 .exactZero (none)

def event3705 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18975⟩⟩) 0 ⟨18637⟩ 3471

def event3706 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18975⟩⟩) (.authority (.programFamilyFact))

def exact3707RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩]

theorem exact3707RawTermsValid :
    exact3707RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3707 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18975⟩⟩) exact3707RawTerms (.finite 3) 3706 .exactZero (none)

def event3708 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18976⟩⟩) 0 ⟨18975⟩ 3707

def event3709 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18976⟩⟩) 1 ⟨6846⟩ 703

def event3710 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18976⟩⟩) (.product (.predecessor 0 3708 .coefficient) (.predecessor 1 3709 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3711 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨18976⟩⟩, .operator (⟨3707, 0⟩, ⟨703, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩)

def exact3712RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩]

theorem exact3712RawTermsValid :
    exact3712RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3712 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18976⟩⟩) exact3712RawTerms (.finite 175932572039110456474905) 3710 .exactZero (none)

def event3713 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16126⟩⟩) 0 ⟨15837⟩ 3494

def event3714 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16126⟩⟩) (.authority (.programFamilyFact))

def exact3715RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩]

theorem exact3715RawTermsValid :
    exact3715RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3715 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16126⟩⟩) exact3715RawTerms (.finite 2) 3714 .exactZero (none)

def event3716 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16127⟩⟩) 0 ⟨16126⟩ 3715

def event3717 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16127⟩⟩) 1 ⟨6863⟩ 713

def event3718 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16127⟩⟩) (.product (.predecessor 0 3716 .coefficient) (.predecessor 1 3717 .coefficient) (⟨true, true, none, some 1, some 1⟩))

def event3719 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨16127⟩⟩, .operator (⟨3715, 0⟩, ⟨713, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩)

def exact3720RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩]

theorem exact3720RawTermsValid :
    exact3720RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3720 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16127⟩⟩) exact3720RawTerms (.finite 156384508479209294644360) 3718 .exactZero (none)

def event3721 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16128⟩⟩) 0 ⟨6728⟩ 728

def event3722 : Event := .predecessor (⟨.program ⟨257⟩, ⟨16128⟩⟩) 1 ⟨16127⟩ 3720

def event3723 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨16128⟩⟩) (.sum [.predecessor 0 3721 .coefficient, .predecessor 1 3722 .coefficient])

def exact3724RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩]

theorem exact3724RawTermsValid :
    exact3724RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3724 : Event := .resultExact (⟨.program ⟨257⟩, ⟨16128⟩⟩) exact3724RawTerms (.finite 156384508479209294644360) 3723 .exactZero (none)

def event3725 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18977⟩⟩) 0 ⟨16128⟩ 3724

def event3726 : Event := .predecessor (⟨.program ⟨257⟩, ⟨18977⟩⟩) 1 ⟨18976⟩ 3712

def event3727 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨18977⟩⟩) (.sum [.predecessor 0 3725 .coefficient, .predecessor 1 3726 .coefficient])

def exact3728RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩]

theorem exact3728RawTermsValid :
    exact3728RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3728 : Event := .resultExact (⟨.program ⟨257⟩, ⟨18977⟩⟩) exact3728RawTerms (.finite 332317080518319751119265) 3727 .exactZero (none)

def event3729 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22197⟩⟩) 0 ⟨18977⟩ 3728

def event3730 : Event := .predecessor (⟨.program ⟨257⟩, ⟨22197⟩⟩) 1 ⟨22196⟩ 3704

def event3731 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨22197⟩⟩) (.sum [.predecessor 0 3729 .coefficient, .predecessor 1 3730 .coefficient])

def exact3732RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩]

theorem exact3732RawTermsValid :
    exact3732RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3732 : Event := .resultExact (⟨.program ⟨257⟩, ⟨22197⟩⟩) exact3732RawTerms (.finite 519978490693370904692497) 3731 .exactZero (none)

def event3733 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32217⟩⟩) 0 ⟨22197⟩ 3732

def event3734 : Event := .predecessor (⟨.program ⟨257⟩, ⟨32217⟩⟩) 1 ⟨32216⟩ 3696

def event3735 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨32217⟩⟩) (.sum [.predecessor 0 3733 .coefficient, .predecessor 1 3734 .coefficient])

def exact3736RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩]

theorem exact3736RawTermsValid :
    exact3736RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3736 : Event := .resultExact (⟨.program ⟨257⟩, ⟨32217⟩⟩) exact3736RawTerms (.finite 721044287309497140663817) 3735 .exactZero (none)

def event3737 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51281⟩⟩) 0 ⟨32217⟩ 3736

def event3738 : Event := .predecessor (⟨.program ⟨257⟩, ⟨51281⟩⟩) 1 ⟨51280⟩ 3688

def event3739 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨51281⟩⟩) (.sum [.predecessor 0 3737 .coefficient, .predecessor 1 3738 .coefficient])

def exact3740RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩]

theorem exact3740RawTermsValid :
    exact3740RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3740 : Event := .resultExact (⟨.program ⟨257⟩, ⟨51281⟩⟩) exact3740RawTerms (.finite 934295889781146178815217) 3739 .exactZero (none)

def event3741 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54261⟩⟩) 0 ⟨51281⟩ 3740

def event3742 : Event := .predecessor (⟨.program ⟨257⟩, ⟨54261⟩⟩) 1 ⟨54260⟩ 3680

def event3743 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨54261⟩⟩) (.sum [.predecessor 0 3741 .coefficient, .predecessor 1 3742 .coefficient])

def exact3744RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩]

theorem exact3744RawTermsValid :
    exact3744RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3744 : Event := .resultExact (⟨.program ⟨257⟩, ⟨54261⟩⟩) exact3744RawTerms (.finite 1150828286136974432938177) 3743 .exactZero (none)

def event3745 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57241⟩⟩) 0 ⟨54261⟩ 3744

def event3746 : Event := .predecessor (⟨.program ⟨257⟩, ⟨57241⟩⟩) 1 ⟨57240⟩ 3672

def event3747 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨57241⟩⟩) (.sum [.predecessor 0 3745 .coefficient, .predecessor 1 3746 .coefficient])

def exact3748RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩]

theorem exact3748RawTermsValid :
    exact3748RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3748 : Event := .resultExact (⟨.program ⟨257⟩, ⟨57241⟩⟩) exact3748RawTerms (.finite 1371606415754681672436097) 3747 .exactZero (none)

def event3749 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60221⟩⟩) 0 ⟨57241⟩ 3748

def event3750 : Event := .predecessor (⟨.program ⟨257⟩, ⟨60221⟩⟩) 1 ⟨60220⟩ 3664

def event3751 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨60221⟩⟩) (.sum [.predecessor 0 3749 .coefficient, .predecessor 1 3750 .coefficient])

def exact3752RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩]

theorem exact3752RawTermsValid :
    exact3752RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3752 : Event := .resultExact (⟨.program ⟨257⟩, ⟨60221⟩⟩) exact3752RawTerms (.finite 1593837033067242249035977) 3751 .exactZero (none)

def event3753 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63201⟩⟩) 0 ⟨60221⟩ 3752

def event3754 : Event := .predecessor (⟨.program ⟨257⟩, ⟨63201⟩⟩) 1 ⟨63200⟩ 3656

def event3755 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨63201⟩⟩) (.sum [.predecessor 0 3753 .coefficient, .predecessor 1 3754 .coefficient])

def exact3756RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩]

theorem exact3756RawTermsValid :
    exact3756RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3756 : Event := .resultExact (⟨.program ⟨257⟩, ⟨63201⟩⟩) exact3756RawTerms (.finite 1818214806102629497873537) 3755 .exactZero (none)

def event3757 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67010⟩⟩) 0 ⟨63201⟩ 3756

def event3758 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67010⟩⟩) 1 ⟨67009⟩ 3648

def event3759 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67010⟩⟩) (.sum [.predecessor 0 3757 .coefficient, .predecessor 1 3758 .coefficient])

def exact3760RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], []⟩, (1)⟩]

theorem exact3760RawTermsValid :
    exact3760RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3760 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67010⟩⟩) exact3760RawTerms (.finite 2044702714934587786668817) 3759 .exactZero (none)

def event3761 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67011⟩⟩) 0 ⟨67010⟩ 3760

def event3762 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67011⟩⟩) 1 ⟨26701⟩ 3640

def event3763 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67011⟩⟩) (.sum [.predecessor 0 3761 .coefficient, .predecessor 1 3762 .coefficient])

def exact3764RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], []⟩, (1)⟩]

theorem exact3764RawTermsValid :
    exact3764RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3764 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67011⟩⟩) exact3764RawTerms (.finite 2271712485307633536959017) 3763 .exactZero (none)

def event3765 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67012⟩⟩) 0 ⟨67011⟩ 3764

def event3766 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67012⟩⟩) 1 ⟨29381⟩ 3632

def event3767 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67012⟩⟩) (.sum [.predecessor 0 3765 .coefficient, .predecessor 1 3766 .coefficient])

def exact3768RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], []⟩, (1)⟩]

theorem exact3768RawTermsValid :
    exact3768RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3768 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67012⟩⟩) exact3768RawTerms (.finite 2499949335520533588602137) 3767 .exactZero (none)

def event3769 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67013⟩⟩) 0 ⟨67012⟩ 3768

def event3770 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67013⟩⟩) 1 ⟨35038⟩ 3624

def event3771 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67013⟩⟩) (.sum [.predecessor 0 3769 .coefficient, .predecessor 1 3770 .coefficient])

def exact3772RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], []⟩, (1)⟩]

theorem exact3772RawTermsValid :
    exact3772RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3772 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67013⟩⟩) exact3772RawTerms (.finite 2728804713782791092959737) 3771 .exactZero (none)

def event3773 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67014⟩⟩) 0 ⟨67013⟩ 3772

def event3774 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67014⟩⟩) 1 ⟨37718⟩ 3616

def event3775 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67014⟩⟩) (.sum [.predecessor 0 3773 .coefficient, .predecessor 1 3774 .coefficient])

def exact3776RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], []⟩, (1)⟩]

theorem exact3776RawTermsValid :
    exact3776RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3776 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67014⟩⟩) exact3776RawTerms (.finite 2957926202950004710694497) 3775 .exactZero (none)

def event3777 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67015⟩⟩) 0 ⟨67014⟩ 3776

def event3778 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67015⟩⟩) 1 ⟨40401⟩ 3608

def event3779 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67015⟩⟩) (.sum [.predecessor 0 3777 .coefficient, .predecessor 1 3778 .coefficient])

def exact3780RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], []⟩, (1)⟩]

theorem exact3780RawTermsValid :
    exact3780RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3780 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67015⟩⟩) exact3780RawTerms (.finite 3187511970717354526236217) 3779 .exactZero (none)

def event3781 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67016⟩⟩) 0 ⟨67015⟩ 3780

def event3782 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67016⟩⟩) 1 ⟨43081⟩ 3600

def event3783 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67016⟩⟩) (.sum [.predecessor 0 3781 .coefficient, .predecessor 1 3782 .coefficient])

def exact3784RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], []⟩, (1)⟩]

theorem exact3784RawTermsValid :
    exact3784RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3784 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67016⟩⟩) exact3784RawTerms (.finite 3417662756781096507033577) 3783 .exactZero (none)

def event3785 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67017⟩⟩) 0 ⟨67016⟩ 3784

def event3786 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67017⟩⟩) 1 ⟨45758⟩ 3592

def event3787 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67017⟩⟩) (.sum [.predecessor 0 3785 .coefficient, .predecessor 1 3786 .coefficient])

def exact3788RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], []⟩, (1)⟩]

theorem exact3788RawTermsValid :
    exact3788RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3788 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67017⟩⟩) exact3788RawTerms (.finite 3648263642165693263543057) 3787 .exactZero (none)

def event3789 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67018⟩⟩) 0 ⟨67017⟩ 3788

def event3790 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67018⟩⟩) 1 ⟨48438⟩ 3584

def event3791 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67018⟩⟩) (.sum [.predecessor 0 3789 .coefficient, .predecessor 1 3790 .coefficient])

def exact3792RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], []⟩, (1)⟩]

theorem exact3792RawTermsValid :
    exact3792RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3792 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67018⟩⟩) exact3792RawTerms (.finite 3878994884184198780231457) 3791 .exactZero (none)

def event3793 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67589⟩⟩) 0 ⟨67018⟩ 3792

def event3794 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67589⟩⟩) 1 ⟨67587⟩ 3576

def event3795 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67589⟩⟩) (.sum [.predecessor 0 3793 .coefficient, .predecessor 1 3794 .coefficient])

def exact3796RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67586⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], []⟩, (1)⟩]

theorem exact3796RawTermsValid :
    exact3796RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3796 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67589⟩⟩) exact3796RawTerms (.finite 8101376613122849735629177) 3795 .exactZero (none)

def event3797 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67590⟩⟩) 0 ⟨67589⟩ 3796

def event3798 : Event := .predecessor (⟨.program ⟨257⟩, ⟨67590⟩⟩) 1 ⟨6733⟩ 3073

def event3799 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨67590⟩⟩) (.product (.predecessor 0 3797 .coefficient) (.predecessor 1 3798 .coefficient) (⟨false, true, none, none, some 1⟩))

def event3800 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67590⟩⟩, .operator (⟨3796, 5⟩, ⟨3073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67586⟩⟩], []⟩, (-1)⟩)

def event3801 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67590⟩⟩, .operator (⟨3796, 7⟩, ⟨3073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], []⟩, (1)⟩)

def event3802 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67590⟩⟩, .operator (⟨3796, 8⟩, ⟨3073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], []⟩, (1)⟩)

def event3803 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67590⟩⟩, .operator (⟨3796, 9⟩, ⟨3073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], []⟩, (1)⟩)

def event3804 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67590⟩⟩, .operator (⟨3796, 11⟩, ⟨3073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], []⟩, (1)⟩)

def event3805 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67590⟩⟩, .operator (⟨3796, 12⟩, ⟨3073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], []⟩, (1)⟩)

def event3806 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67590⟩⟩, .operator (⟨3796, 13⟩, ⟨3073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], []⟩, (1)⟩)

def event3807 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67590⟩⟩, .operator (⟨3796, 15⟩, ⟨3073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], []⟩, (1)⟩)

def event3808 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67590⟩⟩, .operator (⟨3796, 16⟩, ⟨3073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], []⟩, (1)⟩)

def event3809 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67590⟩⟩, .operator (⟨3796, 18⟩, ⟨3073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], []⟩, (1)⟩)

def event3810 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67590⟩⟩, .operator (⟨3796, 0⟩, ⟨3073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], []⟩, (1)⟩)

def event3811 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67590⟩⟩, .operator (⟨3796, 1⟩, ⟨3073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], []⟩, (1)⟩)

def event3812 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67590⟩⟩, .operator (⟨3796, 2⟩, ⟨3073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], []⟩, (1)⟩)

def event3813 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67590⟩⟩, .operator (⟨3796, 3⟩, ⟨3073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], []⟩, (1)⟩)

def event3814 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67590⟩⟩, .operator (⟨3796, 4⟩, ⟨3073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩, (1)⟩)

def event3815 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67590⟩⟩, .operator (⟨3796, 6⟩, ⟨3073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩, (1)⟩)

def event3816 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67590⟩⟩, .operator (⟨3796, 10⟩, ⟨3073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩)

def event3817 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67590⟩⟩, .operator (⟨3796, 14⟩, ⟨3073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩)

def event3818 : Event := .coefficientMerge (⟨⟨.program ⟨257⟩, ⟨67590⟩⟩, .operator (⟨3796, 17⟩, ⟨3073, 0⟩), ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩)

def exact3819RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67586⟩⟩], []⟩, (-1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩, (1)⟩, ⟨⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], []⟩, (1)⟩]

theorem exact3819RawTermsValid :
    exact3819RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3819 : Event := .resultExact (⟨.program ⟨257⟩, ⟨67590⟩⟩) exact3819RawTerms (.finite 288437114210242386499449054215782066723695848195610074846773075121806196758232336412366319954652507496327746424744120105673250933066474317800169869469972561176405065563016300397404185154408596701184) 3799 .exactZero (none)

def event3820 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨6755⟩⟩) (.authority (.factStore))

def exact3821RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨6755⟩⟩], []⟩, (1)⟩]

theorem exact3821RawTermsValid :
    exact3821RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3821 : Event := .resultExact (⟨.program ⟨257⟩, ⟨6755⟩⟩) exact3821RawTerms (.finite 842028614180002698769518329474413554821484498062532544955551025947591147239147143367264087617405288658407746022228090406510614415879737123091517018472577532187540201282) 3820 .exactZero (none)

def event3822 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.authority (.operator))

def event3823 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9843⟩⟩) (.finite 14)

def event3824 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 0 ⟨392⟩ 14

def event3825 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9845⟩⟩) 1 ⟨9843⟩ 3823

def event3826 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.sum [.predecessor 0 3824 .coefficient, .predecessor 1 3825 .coefficient])

def event3827 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9845⟩⟩) (.finite 655354)

def event3828 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 0 ⟨9845⟩ 3827

def event3829 : Event := .predecessor (⟨.program ⟨257⟩, ⟨9901⟩⟩) 1 ⟨5426⟩ 38

def event3830 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.identity (.predecessor 1 3829 .coefficient))

def event3831 : Event := .resultCoefficient (⟨.program ⟨257⟩, ⟨9901⟩⟩) (.finite 655360)

def event3832 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47954⟩⟩) 0 ⟨9901⟩ 3831

def event3833 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨47954⟩⟩) (.authority (.programFamilyFact))

def exact3834RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨47954⟩⟩], []⟩, (1)⟩]

theorem exact3834RawTermsValid :
    exact3834RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3834 : Event := .resultExact (⟨.program ⟨257⟩, ⟨47954⟩⟩) exact3834RawTerms (.finite 60) 3833 .exactZero (none)

def event3835 : Event := .predecessor (⟨.program ⟨257⟩, ⟨15156⟩⟩) 0 ⟨9901⟩ 3831

def event3836 : Event := .boundTransfer (⟨.program ⟨257⟩, ⟨15156⟩⟩) (.authority (.programFamilyFact))

def exact3837RawTerms : List Term := [⟨⟨[⟨.program ⟨257⟩, ⟨15156⟩⟩], []⟩, (1)⟩]

theorem exact3837RawTermsValid :
    exact3837RawTerms.all (fun term => monomialValid document term.monomial) = true := by
  decide +kernel

def event3837 : Event := .resultExact (⟨.program ⟨257⟩, ⟨15156⟩⟩) exact3837RawTerms (.finite 60) 3836 .exactZero (none)

def event3838 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47955⟩⟩) 0 ⟨15156⟩ 3837

def event3839 : Event := .predecessor (⟨.program ⟨257⟩, ⟨47955⟩⟩) 1 ⟨47954⟩ 3834

def eventLeaf224 : Array AnnotatedEvent := #[
  { event := event3584
    frameStart := 0 },
  { event := event3585
    frameStart := 0 },
  { event := event3586
    frameStart := 0 },
  { event := event3587
    frameStart := 0 },
  { event := event3588
    frameStart := 0 },
  { event := event3589
    frameStart := 0 },
  { event := event3590
    frameStart := 0 },
  { event := event3591
    frameStart := 0 },
  { event := event3592
    frameStart := 0 },
  { event := event3593
    frameStart := 0 },
  { event := event3594
    frameStart := 0 },
  { event := event3595
    frameStart := 0 },
  { event := event3596
    frameStart := 0 },
  { event := event3597
    frameStart := 0 },
  { event := event3598
    frameStart := 0 },
  { event := event3599
    frameStart := 0 }
]

def eventLeaf225 : Array AnnotatedEvent := #[
  { event := event3600
    frameStart := 0 },
  { event := event3601
    frameStart := 0 },
  { event := event3602
    frameStart := 0 },
  { event := event3603
    frameStart := 0 },
  { event := event3604
    frameStart := 0 },
  { event := event3605
    frameStart := 0 },
  { event := event3606
    frameStart := 0 },
  { event := event3607
    frameStart := 0 },
  { event := event3608
    frameStart := 0 },
  { event := event3609
    frameStart := 0 },
  { event := event3610
    frameStart := 0 },
  { event := event3611
    frameStart := 0 },
  { event := event3612
    frameStart := 0 },
  { event := event3613
    frameStart := 0 },
  { event := event3614
    frameStart := 0 },
  { event := event3615
    frameStart := 0 }
]

def eventLeaf226 : Array AnnotatedEvent := #[
  { event := event3616
    frameStart := 0 },
  { event := event3617
    frameStart := 0 },
  { event := event3618
    frameStart := 0 },
  { event := event3619
    frameStart := 0 },
  { event := event3620
    frameStart := 0 },
  { event := event3621
    frameStart := 0 },
  { event := event3622
    frameStart := 0 },
  { event := event3623
    frameStart := 0 },
  { event := event3624
    frameStart := 0 },
  { event := event3625
    frameStart := 0 },
  { event := event3626
    frameStart := 0 },
  { event := event3627
    frameStart := 0 },
  { event := event3628
    frameStart := 0 },
  { event := event3629
    frameStart := 0 },
  { event := event3630
    frameStart := 0 },
  { event := event3631
    frameStart := 0 }
]

def eventLeaf227 : Array AnnotatedEvent := #[
  { event := event3632
    frameStart := 0 },
  { event := event3633
    frameStart := 0 },
  { event := event3634
    frameStart := 0 },
  { event := event3635
    frameStart := 0 },
  { event := event3636
    frameStart := 0 },
  { event := event3637
    frameStart := 0 },
  { event := event3638
    frameStart := 0 },
  { event := event3639
    frameStart := 0 },
  { event := event3640
    frameStart := 0 },
  { event := event3641
    frameStart := 0 },
  { event := event3642
    frameStart := 0 },
  { event := event3643
    frameStart := 0 },
  { event := event3644
    frameStart := 0 },
  { event := event3645
    frameStart := 0 },
  { event := event3646
    frameStart := 0 },
  { event := event3647
    frameStart := 0 }
]

def eventLeaf228 : Array AnnotatedEvent := #[
  { event := event3648
    frameStart := 0 },
  { event := event3649
    frameStart := 0 },
  { event := event3650
    frameStart := 0 },
  { event := event3651
    frameStart := 0 },
  { event := event3652
    frameStart := 0 },
  { event := event3653
    frameStart := 0 },
  { event := event3654
    frameStart := 0 },
  { event := event3655
    frameStart := 0 },
  { event := event3656
    frameStart := 0 },
  { event := event3657
    frameStart := 0 },
  { event := event3658
    frameStart := 0 },
  { event := event3659
    frameStart := 0 },
  { event := event3660
    frameStart := 0 },
  { event := event3661
    frameStart := 0 },
  { event := event3662
    frameStart := 0 },
  { event := event3663
    frameStart := 0 }
]

def eventLeaf229 : Array AnnotatedEvent := #[
  { event := event3664
    frameStart := 0 },
  { event := event3665
    frameStart := 0 },
  { event := event3666
    frameStart := 0 },
  { event := event3667
    frameStart := 0 },
  { event := event3668
    frameStart := 0 },
  { event := event3669
    frameStart := 0 },
  { event := event3670
    frameStart := 0 },
  { event := event3671
    frameStart := 0 },
  { event := event3672
    frameStart := 0 },
  { event := event3673
    frameStart := 0 },
  { event := event3674
    frameStart := 0 },
  { event := event3675
    frameStart := 0 },
  { event := event3676
    frameStart := 0 },
  { event := event3677
    frameStart := 0 },
  { event := event3678
    frameStart := 0 },
  { event := event3679
    frameStart := 0 }
]

def eventLeaf230 : Array AnnotatedEvent := #[
  { event := event3680
    frameStart := 0 },
  { event := event3681
    frameStart := 0 },
  { event := event3682
    frameStart := 0 },
  { event := event3683
    frameStart := 0 },
  { event := event3684
    frameStart := 0 },
  { event := event3685
    frameStart := 0 },
  { event := event3686
    frameStart := 0 },
  { event := event3687
    frameStart := 0 },
  { event := event3688
    frameStart := 0 },
  { event := event3689
    frameStart := 0 },
  { event := event3690
    frameStart := 0 },
  { event := event3691
    frameStart := 0 },
  { event := event3692
    frameStart := 0 },
  { event := event3693
    frameStart := 0 },
  { event := event3694
    frameStart := 0 },
  { event := event3695
    frameStart := 0 }
]

def eventLeaf231 : Array AnnotatedEvent := #[
  { event := event3696
    frameStart := 0 },
  { event := event3697
    frameStart := 0 },
  { event := event3698
    frameStart := 0 },
  { event := event3699
    frameStart := 0 },
  { event := event3700
    frameStart := 0 },
  { event := event3701
    frameStart := 0 },
  { event := event3702
    frameStart := 0 },
  { event := event3703
    frameStart := 0 },
  { event := event3704
    frameStart := 0 },
  { event := event3705
    frameStart := 0 },
  { event := event3706
    frameStart := 0 },
  { event := event3707
    frameStart := 0 },
  { event := event3708
    frameStart := 0 },
  { event := event3709
    frameStart := 0 },
  { event := event3710
    frameStart := 0 },
  { event := event3711
    frameStart := 0 }
]

def eventLeaf232 : Array AnnotatedEvent := #[
  { event := event3712
    frameStart := 0 },
  { event := event3713
    frameStart := 0 },
  { event := event3714
    frameStart := 0 },
  { event := event3715
    frameStart := 0 },
  { event := event3716
    frameStart := 0 },
  { event := event3717
    frameStart := 0 },
  { event := event3718
    frameStart := 0 },
  { event := event3719
    frameStart := 0 },
  { event := event3720
    frameStart := 0 },
  { event := event3721
    frameStart := 0 },
  { event := event3722
    frameStart := 0 },
  { event := event3723
    frameStart := 0 },
  { event := event3724
    frameStart := 0 },
  { event := event3725
    frameStart := 0 },
  { event := event3726
    frameStart := 0 },
  { event := event3727
    frameStart := 0 }
]

def eventLeaf233 : Array AnnotatedEvent := #[
  { event := event3728
    frameStart := 0 },
  { event := event3729
    frameStart := 0 },
  { event := event3730
    frameStart := 0 },
  { event := event3731
    frameStart := 0 },
  { event := event3732
    frameStart := 0 },
  { event := event3733
    frameStart := 0 },
  { event := event3734
    frameStart := 0 },
  { event := event3735
    frameStart := 0 },
  { event := event3736
    frameStart := 0 },
  { event := event3737
    frameStart := 0 },
  { event := event3738
    frameStart := 0 },
  { event := event3739
    frameStart := 0 },
  { event := event3740
    frameStart := 0 },
  { event := event3741
    frameStart := 0 },
  { event := event3742
    frameStart := 0 },
  { event := event3743
    frameStart := 0 }
]

def eventLeaf234 : Array AnnotatedEvent := #[
  { event := event3744
    frameStart := 0 },
  { event := event3745
    frameStart := 0 },
  { event := event3746
    frameStart := 0 },
  { event := event3747
    frameStart := 0 },
  { event := event3748
    frameStart := 0 },
  { event := event3749
    frameStart := 0 },
  { event := event3750
    frameStart := 0 },
  { event := event3751
    frameStart := 0 },
  { event := event3752
    frameStart := 0 },
  { event := event3753
    frameStart := 0 },
  { event := event3754
    frameStart := 0 },
  { event := event3755
    frameStart := 0 },
  { event := event3756
    frameStart := 0 },
  { event := event3757
    frameStart := 0 },
  { event := event3758
    frameStart := 0 },
  { event := event3759
    frameStart := 0 }
]

def eventLeaf235 : Array AnnotatedEvent := #[
  { event := event3760
    frameStart := 0 },
  { event := event3761
    frameStart := 0 },
  { event := event3762
    frameStart := 0 },
  { event := event3763
    frameStart := 0 },
  { event := event3764
    frameStart := 0 },
  { event := event3765
    frameStart := 0 },
  { event := event3766
    frameStart := 0 },
  { event := event3767
    frameStart := 0 },
  { event := event3768
    frameStart := 0 },
  { event := event3769
    frameStart := 0 },
  { event := event3770
    frameStart := 0 },
  { event := event3771
    frameStart := 0 },
  { event := event3772
    frameStart := 0 },
  { event := event3773
    frameStart := 0 },
  { event := event3774
    frameStart := 0 },
  { event := event3775
    frameStart := 0 }
]

def eventLeaf236 : Array AnnotatedEvent := #[
  { event := event3776
    frameStart := 0 },
  { event := event3777
    frameStart := 0 },
  { event := event3778
    frameStart := 0 },
  { event := event3779
    frameStart := 0 },
  { event := event3780
    frameStart := 0 },
  { event := event3781
    frameStart := 0 },
  { event := event3782
    frameStart := 0 },
  { event := event3783
    frameStart := 0 },
  { event := event3784
    frameStart := 0 },
  { event := event3785
    frameStart := 0 },
  { event := event3786
    frameStart := 0 },
  { event := event3787
    frameStart := 0 },
  { event := event3788
    frameStart := 0 },
  { event := event3789
    frameStart := 0 },
  { event := event3790
    frameStart := 0 },
  { event := event3791
    frameStart := 0 }
]

def eventLeaf237 : Array AnnotatedEvent := #[
  { event := event3792
    frameStart := 0 },
  { event := event3793
    frameStart := 0 },
  { event := event3794
    frameStart := 0 },
  { event := event3795
    frameStart := 0 },
  { event := event3796
    frameStart := 0 },
  { event := event3797
    frameStart := 0 },
  { event := event3798
    frameStart := 0 },
  { event := event3799
    frameStart := 0 },
  { event := event3800
    frameStart := 0 },
  { event := event3801
    frameStart := 0 },
  { event := event3802
    frameStart := 0 },
  { event := event3803
    frameStart := 0 },
  { event := event3804
    frameStart := 0 },
  { event := event3805
    frameStart := 0 },
  { event := event3806
    frameStart := 0 },
  { event := event3807
    frameStart := 0 }
]

def eventLeaf238 : Array AnnotatedEvent := #[
  { event := event3808
    frameStart := 0 },
  { event := event3809
    frameStart := 0 },
  { event := event3810
    frameStart := 0 },
  { event := event3811
    frameStart := 0 },
  { event := event3812
    frameStart := 0 },
  { event := event3813
    frameStart := 0 },
  { event := event3814
    frameStart := 0 },
  { event := event3815
    frameStart := 0 },
  { event := event3816
    frameStart := 0 },
  { event := event3817
    frameStart := 0 },
  { event := event3818
    frameStart := 0 },
  { event := event3819
    frameStart := 0 },
  { event := event3820
    frameStart := 0 },
  { event := event3821
    frameStart := 0 },
  { event := event3822
    frameStart := 0 },
  { event := event3823
    frameStart := 0 }
]

def eventLeaf239 : Array AnnotatedEvent := #[
  { event := event3824
    frameStart := 0 },
  { event := event3825
    frameStart := 0 },
  { event := event3826
    frameStart := 0 },
  { event := event3827
    frameStart := 0 },
  { event := event3828
    frameStart := 0 },
  { event := event3829
    frameStart := 0 },
  { event := event3830
    frameStart := 0 },
  { event := event3831
    frameStart := 0 },
  { event := event3832
    frameStart := 0 },
  { event := event3833
    frameStart := 0 },
  { event := event3834
    frameStart := 0 },
  { event := event3835
    frameStart := 0 },
  { event := event3836
    frameStart := 0 },
  { event := event3837
    frameStart := 0 },
  { event := event3838
    frameStart := 0 },
  { event := event3839
    frameStart := 0 }
]

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.Events014
