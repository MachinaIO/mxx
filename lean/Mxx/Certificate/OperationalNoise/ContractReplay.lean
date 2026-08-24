import Mxx.Certificate.OperationalNoise.OperatorReplay

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.EventReplay

/-- Convert one explicitly recorded finite maximum exactly as Rust does: zero is exact zero and
    a positive maximum carries the finite-class invariant. -/
def recordedFiniteContract : Nat → CoeffClass
  | 0 => .exactZero
  | maximum + 1 => .finite ⟨maximum + 1, Nat.zero_lt_succ maximum⟩

@[simp]
theorem recordedFiniteContract_zero : recordedFiniteContract 0 = .exactZero := by
  rfl

@[simp]
theorem recordedFiniteContract_succ (maximum : Nat) :
    recordedFiniteContract (maximum + 1) =
      .finite ⟨maximum + 1, Nat.zero_lt_succ maximum⟩ := by
  rfl

theorem recordedFiniteContract_sound {maximum actual : Nat} (actualBound : actual ≤ maximum) :
    (recordedFiniteContract maximum).Interprets actual := by
  cases maximum with
  | zero => simp_all [recordedFiniteContract, CoeffClass.Interprets]
  | succ maximum =>
      exact actualBound

/-- The maximum absolute endpoint recorded for a signed uniform interval. -/
def uniformIntervalMaximum (minimum maximum : Int) : Nat :=
  Nat.max minimum.natAbs maximum.natAbs

/-- The exact reached Rust interval branch. Empty intervals are rejected; valid intervals at or
    beyond the centered halfway point are `large`, not a modulus-sized finite fallback. -/
def uniformIntervalContract (modulus : Nat) (minimum maximum : Int) : Option CoeffClass :=
  if minimum > maximum then none
  else
    let upper := uniformIntervalMaximum minimum maximum
    if 2 * upper >= modulus then some .large else some (recordedFiniteContract upper)

theorem interval_natAbs_le_maximum {minimum value maximum : Int}
    (valueInRange : minimum ≤ value ∧ value ≤ maximum) :
    value.natAbs ≤ uniformIntervalMaximum minimum maximum := by
  unfold uniformIntervalMaximum
  by_cases valueNonnegative : 0 ≤ value
  · apply Nat.le_trans _ (Nat.le_max_right _ _)
    rw [← Int.ofNat_le, Int.ofNat_natAbs_of_nonneg valueNonnegative]
    exact Int.le_trans valueInRange.2 Int.le_natAbs
  · have valueNonpositive : value ≤ 0 := by omega
    apply Nat.le_trans _ (Nat.le_max_left _ _)
    rw [← Int.ofNat_le, Int.ofNat_natAbs_of_nonpos valueNonpositive]
    exact Int.le_trans (Int.neg_le_neg valueInRange.1) Lean.Omega.Int.neg_le_natAbs

theorem uniformIntervalContract_sound {modulus : Nat} {minimum value maximum : Int}
    {bound : CoeffClass}
    (contract : uniformIntervalContract modulus minimum maximum = some bound)
    (valueInRange : minimum ≤ value ∧ value ≤ maximum) :
    bound.Interprets value.natAbs := by
  unfold uniformIntervalContract at contract
  split at contract
  · contradiction
  · dsimp only at contract
    split at contract
    · cases contract
      trivial
    · cases contract
      exact recordedFiniteContract_sound (interval_natAbs_le_maximum valueInRange)

theorem uniformIntervalContract_finite {modulus : Nat} {minimum maximum : Int}
    (validRange : minimum ≤ maximum)
    (insideCenteredHalf : 2 * uniformIntervalMaximum minimum maximum < modulus) :
    uniformIntervalContract modulus minimum maximum =
      some (recordedFiniteContract (uniformIntervalMaximum minimum maximum)) := by
  simp [uniformIntervalContract, Int.not_lt.mpr validRange,
    Nat.not_le.mpr insideCenteredHalf]

/-- Gaussian cutoff replay consumes the recorded cutoff and a coefficient-side bound premise. -/
theorem gaussianCutoff_sound {cutoff actual : Nat} (actualBound : actual ≤ cutoff) :
    (recordedFiniteContract cutoff).Interprets actual :=
  recordedFiniteContract_sound actualBound

/-- Preimage cutoff replay consumes its own recorded cutoff; trapdoor metadata is not accepted. -/
theorem preimageCutoff_sound {cutoff actual : Nat} (actualBound : actual ≤ cutoff) :
    (recordedFiniteContract cutoff).Interprets actual :=
  recordedFiniteContract_sound actualBound

/-- The matrix-valued trapdoor sampler is the public uniform matrix, hence `large`. -/
def trapdoorSamplerContract : CoeffClass := .large

theorem trapdoorSamplerContract_sound (actual : Nat) :
    trapdoorSamplerContract.Interprets actual := by
  trivial

def regularGadgetMaximum (base : Nat) : Nat := Nat.max (base / 2) 1

theorem regularGadgetMaximum_positive (base : Nat) : 0 < regularGadgetMaximum base := by
  exact Nat.lt_of_lt_of_le Nat.zero_lt_one (Nat.le_max_right _ _)

/-- Regular gadget decomposition is finite exactly for a valid base. -/
def regularGadgetContract (base : Nat) : Option CoeffClass :=
  if base < 2 then none
  else some (.finite ⟨regularGadgetMaximum base, regularGadgetMaximum_positive base⟩)

theorem regularGadgetContract_sound {base : Nat} {digit : Int}
    (baseValid : 2 ≤ base) (digitBound : digit.natAbs ≤ regularGadgetMaximum base) :
    regularGadgetContract base =
        some (.finite ⟨regularGadgetMaximum base, regularGadgetMaximum_positive base⟩) ∧
      (CoeffClass.finite ⟨regularGadgetMaximum base,
        regularGadgetMaximum_positive base⟩).Interprets digit.natAbs := by
  exact ⟨by simp [regularGadgetContract, Nat.not_lt.mpr baseValid], digitBound⟩

/-- A FactStore authority consumes a present typed contract and its semantic witness. -/
theorem factStoreAuthority_sound {contract : Option CoeffClass} {recorded : CoeffClass}
    {actual : Nat} (contractPresent : contract = some recorded)
    (recordedSound : recorded.Interprets actual) : recorded.Interprets actual := by
  cases contractPresent
  exact recordedSound

/-- A program-family authority consumes the present contract attached to that family. -/
theorem programFamilyFactAuthority_sound {contract : Option CoeffClass}
    {recorded : CoeffClass} {actual : Nat} (contractPresent : contract = some recorded)
    (recordedSound : recorded.Interprets actual) : recorded.Interprets actual := by
  cases contractPresent
  exact recordedSound

/-- A relation-preimage authority consumes the matched source's explicit cutoff contract. -/
theorem relationPreimageSourceAuthority_sound {sourceCutoff : Option Nat} {cutoff actual : Nat}
    (cutoffPresent : sourceCutoff = some cutoff) (actualWithinSourceCutoff : actual ≤ cutoff) :
    (recordedFiniteContract cutoff).Interprets actual := by
  cases cutoffPresent
  exact preimageCutoff_sound actualWithinSourceCutoff

end Mxx.Certificate.OperationalNoise.EventReplay
