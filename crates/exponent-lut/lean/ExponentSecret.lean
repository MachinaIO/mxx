import ExponentPairedSwitch

namespace MxxExponentLut

open Mxx.Primitives

/-- The actual two-column secret constructor supplies the integer lift and
small-coefficient premises used by paired switching. The smallness premise is
extracted from the interval sampler's integer witness. -/
theorem augmented_secret_of_sample {q n : Nat}
    (small unit : ExactMatrix q n 1 1) (source : ExactMatrix q n 1 2)
    (sample : MxxRuntime.uniformIntervalSample (-1) 1 small)
    (concat : MxxRuntime.concatColumns small unit source)
    (unit00 : unit 0 0 = 1) :
    ∃ secret : ErrorMatrix n 1 2,
      reduceMatrix q n 1 2 secret = source ∧
      (∀ i, ((secret 0 0).coeff i).natAbs ≤ 1) ∧ secret 0 1 = 1 := by
  obtain ⟨_, sampled, hsampled, hinterval⟩ := sample
  let secret : ErrorMatrix n 1 2 := fun _ column => if column = 0 then sampled 0 0 else 1
  refine ⟨secret, ?_, ?_, ?_⟩
  · ext row column
    fin_cases row
    fin_cases column
    · have h := concat 0 0
      have hs := congrArg (fun matrix : ExactMatrix q n 1 1 => matrix 0 0) hsampled
      simpa [MxxRuntime.concatColumns, secret, reduceMatrix, hs] using h.symm
    · have h := concat 0 1
      simpa [MxxRuntime.concatColumns, secret, reduceMatrix, unit00] using h.symm
  · intro i
    have hi := hinterval 0 0 i
    change ((sampled 0 0).coeff i).natAbs ≤ 1
    omega
  · simp [secret]

end MxxExponentLut
