import ExponentTransfers
import RuntimeMatrixOps
import RuntimeDecomposition

namespace MxxExponentLut
open Mxx.Primitives MxxRuntime
open scoped BigOperators

/-- The fixed RHS producer's two-row concatenation cancels the sampled top
matrix against its explicit last row. Both slice and concat premises are the
actual exported runtime relations from `encoding.rs::sample_fixed_rhs`. -/
theorem sample_fixed_rhs_equation {q n columns : Nat}
    {source target : ExactMatrix q n 1 2} {sourcePrefix : ExactMatrix q n 1 1}
    {gadget ciphertext : ExactMatrix q n 2 columns}
    {top last error : ExactMatrix q n 1 columns} {payload : ExactPoly q n}
    (hsource : source 0 1 = 1)
    (hslice : sliceMatrix source 0 1 0 1 sourcePrefix)
    (hlast : last = payload • (target * gadget) - sourcePrefix * top + error)
    (hconcat : concatRows top last ciphertext) :
    source * ciphertext = payload • (target * gadget) + error := by
  have hp : sourcePrefix 0 0 = source 0 0 := by
    simpa using hslice.2.2.2.2.2.2.2.2 0 0 (by decide) (by decide)
  have htop (column : Fin columns) : ciphertext 0 column = top 0 column := by
    simpa [concatRows] using hconcat 0 column
  have hbottom (column : Fin columns) : ciphertext 1 column = last 0 column := by
    simpa [concatRows] using hconcat 1 column
  ext row column
  have hr : row = 0 := Subsingleton.elim _ _
  subst row
  simp only [Matrix.mul_apply, Fin.sum_univ_two, htop, hbottom, hsource, one_mul]
  rw [hlast]
  simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul,
    Matrix.mul_apply, Fin.sum_univ_one, hp]
  ring

/-- The appended scalar identity establishes the producer's augmented-secret
last coordinate directly from the exported column-concatenation relation. -/
theorem augmented_source_last {q n : Nat}
    {secret unit : ExactMatrix q n 1 1} {source : ExactMatrix q n 1 2}
    (hconcat : concatColumns secret unit source) (hunit : unit 0 0 = 1) :
    source 0 1 = 1 := by
  have h := hconcat 0 1
  simpa [hunit] using h

/-- Gaussian execution exposes the sampled integer witness and its exact cutoff. -/
theorem gaussian_error_witness {q n rows columns : Nat} {sigma : Rat} {cutoff : Int}
    {sample : ExactMatrix q n rows columns} (hrun : gaussianSample sigma cutoff sample) :
    ∃ error : ErrorMatrix n rows columns,
      sample = reduceMatrix q n rows columns error ∧ CoeffBound error cutoff.toNat :=
  hrun.2.2.1

/-- The producer and cached decomposition yield the private-action equation
with a sampled integer error, rather than an assumed cached error equation. -/
theorem sampled_fixed_rhs_cache {q n columns inner : Nat}
    (hq : 1 < q) (hn : 0 < n) {backend : BackendContext} {base digits : Int}
    {sigma : Rat} {cutoff : Int}
    {source target : ExactMatrix q n 1 2} {sourcePrefix : ExactMatrix q n 1 1}
    {targetGadget ciphertext : ExactMatrix q n 2 columns}
    {cacheGadget : ExactMatrix q n 2 inner} {decomposition : ExactMatrix q n inner columns}
    {top last sample : ExactMatrix q n 1 columns} {payload : ExactPoly q n}
    (hsource : source 0 1 = 1)
    (hslice : sliceMatrix source 0 1 0 1 sourcePrefix)
    (hlast : last = payload • (target * targetGadget) - sourcePrefix * top + sample)
    (hconcat : concatRows top last ciphertext)
    (hgaussian : gaussianSample sigma cutoff sample)
    (hgadget : gadgetMatrixRuns backend base digits cacheGadget)
    (hdecomposition : gadgetDecomposeRuns backend base digits ciphertext decomposition)
    (hexact : (backend.regularLayout q n).map RegularLayout.droppedModuli = some 0) :
    ∃ error : ErrorMatrix n 1 columns,
      source * cacheGadget * decomposition = payload • (target * targetGadget) +
        reduceMatrix q n 1 columns error ∧ CoeffBound error cutoff.toNat := by
  obtain ⟨error, he, hb⟩ := gaussian_error_witness hgaussian
  refine ⟨error, ?_, hb⟩
  rw [Matrix.mul_assoc, gadget_runs_reconstruct hq hn hgadget hdecomposition hexact]
  rw [sample_fixed_rhs_equation hsource hslice hlast hconcat, he]

/-- Approximate caches retain the same sampled error and the secret-weighted
decomposition residual. No independence or secret norm is assumed here. -/
theorem sampled_fixed_rhs_cache_residual {q n columns inner : Nat}
    {backend : BackendContext} {base digits : Int} {sigma : Rat} {cutoff : Int}
    {source target : ExactMatrix q n 1 2} {sourcePrefix : ExactMatrix q n 1 1}
    {targetGadget ciphertext : ExactMatrix q n 2 columns}
    {cacheGadget : ExactMatrix q n 2 inner} {decomposition : ExactMatrix q n inner columns}
    {top last sample : ExactMatrix q n 1 columns} {payload : ExactPoly q n}
    (hsource : source 0 1 = 1)
    (hslice : sliceMatrix source 0 1 0 1 sourcePrefix)
    (hlast : last = payload • (target * targetGadget) - sourcePrefix * top + sample)
    (hconcat : concatRows top last ciphertext)
    (hgaussian : gaussianSample sigma cutoff sample)
    (hgadget : gadgetMatrixRuns backend base digits cacheGadget)
    (hdecomposition : gadgetDecomposeRuns backend base digits ciphertext decomposition) :
    ∃ layout : RegularLayout q, ∃ error : ErrorMatrix n 1 columns,
      ∃ residual : ErrorMatrix n 2 columns,
      backend.regularLayout q n = some layout ∧
      CoeffBound error cutoff.toNat ∧ CoeffBound residual layout.errorBound ∧
      source * cacheGadget * decomposition = payload • (target * targetGadget) +
        reduceMatrix q n 1 columns error - source * reduceMatrix q n 2 columns residual := by
  obtain ⟨layout, lookup, residual, hr, hb⟩ :=
    gadgetDecomposeRuns_approximate hgadget hdecomposition
  obtain ⟨error, he, heBound⟩ := gaussian_error_witness hgaussian
  refine ⟨layout, error, residual, lookup, heBound, hb, ?_⟩
  have reconstruction : cacheGadget * decomposition =
      ciphertext - reduceMatrix q n 2 columns residual :=
    eq_sub_iff_add_eq.mpr hr.symm
  rw [Matrix.mul_assoc, reconstruction, Matrix.mul_sub,
    sample_fixed_rhs_equation hsource hslice hlast hconcat, he]

/-- Compose a sampled selector equation with its actual decomposition run.
The integer action gain comes from that run's retained digit layout; the
sampled error is reused unchanged, without a new error or independence premise.
This lemma is independent of whether selection and decomposition share a loop. -/
theorem sampled_selector_cache {q n columns : Nat}
    (hq : 1 < q) (hn : 0 < n)
    {backend : BackendContext} {base digits : Int} {cutoff : Nat}
    {secret : ExactMatrix q n 1 2} {payload : ExactPoly q n}
    {ciphertext : ExactMatrix q n 2 columns}
    {decomposition : ExactMatrix q n columns columns}
    (layout : RegularLayout q)
    (lookup : backend.regularLayout q n = some layout)
    (sampled : ∃ gadget : ExactMatrix q n 2 columns,
      ∃ error : ErrorMatrix n 1 columns,
        gadgetMatrixRuns backend base digits gadget ∧
        secret * ciphertext = payload • (secret * gadget) +
          reduceMatrix q n 1 columns error ∧
        CoeffBound error cutoff)
    (decomposed : gadgetDecomposeRuns backend base digits ciphertext decomposition)
    (exactLayout : (backend.regularLayout q n).map RegularLayout.droppedModuli = some 0) :
    ∃ gadget : ExactMatrix q n 2 columns, ∃ error : ErrorMatrix n 1 columns,
    ∃ action : ErrorMatrix n columns columns,
      gadgetMatrixRuns backend base digits gadget ∧
      decomposition = reduceMatrix q n columns columns action ∧
      ActionGain action (denseRegularGain layout n 2) ∧
      secret * gadget * decomposition = payload • (secret * gadget) +
        reduceMatrix q n 1 columns error ∧
      CoeffBound error cutoff := by
  obtain ⟨gadget, error, hg, he, hb⟩ := sampled
  obtain ⟨action, ha, hgain⟩ :=
    gadgetDecomposeRuns_dense_gain layout lookup hq hn decomposed
  refine ⟨gadget, error, action, hg, ha, hgain, ?_, hb⟩
  rw [Matrix.mul_assoc, gadget_runs_reconstruct hq hn hg decomposed exactLayout]
  exact he

/-- Apply one sampled selector cache to an encoding using the executed product.
The error and action witnesses are obtained from the fixed cache lemma above. -/
theorem sampled_selector_private_step {q n columns : Nat}
    (hq : 1 < q) (hn : 0 < n)
    {backend : BackendContext} {base digits : Int} {cutoff inputBound : Nat}
    {secret mask : ExactMatrix q n 1 2} {payload message : ExactPoly q n}
    {ciphertext gadget publicKey : ExactMatrix q n 2 columns}
    {decomposition : ExactMatrix q n columns columns}
    {input output : ExactMatrix q n 1 columns}
    {inputError : ErrorMatrix n 1 columns} {messageInteger : ErrorPoly n}
    (layout : RegularLayout q)
    (lookup : backend.regularLayout q n = some layout)
    (sampled : ∃ sampledGadget : ExactMatrix q n 2 columns,
      ∃ error : ErrorMatrix n 1 columns,
        gadgetMatrixRuns backend base digits sampledGadget ∧
        secret * ciphertext = payload • (secret * sampledGadget) +
          reduceMatrix q n 1 columns error ∧
        CoeffBound error cutoff)
    (gadgetRuns : gadgetMatrixRuns backend base digits gadget)
    (decomposed : gadgetDecomposeRuns backend base digits ciphertext decomposition)
    (online : output = input * decomposition)
    (encoded : input = mask * publicKey - message • (secret * gadget) +
      reduceMatrix q n 1 columns inputError)
    (messageLift : message = reducePoly q n messageInteger)
    (messageBound : polynomialL1 messageInteger ≤ 1)
    (inputBounded : CoeffBound inputError inputBound)
    (exactLayout : (backend.regularLayout q n).map RegularLayout.droppedModuli = some 0) :
    Approx output (mask * (publicKey * decomposition) -
      (message * payload) • (secret * gadget))
      (denseRegularGain layout n 2 * inputBound + cutoff) := by
  obtain ⟨other, error, action, hg, ha, hgain, hcache, hb⟩ :=
    sampled_selector_cache hq hn layout lookup sampled decomposed exactLayout
  have same : other = gadget := gadget_runs_deterministic hg gadgetRuns
  subst other
  rw [ha] at hcache
  rw [online, ha]
  exact private_encoding_transfer_sampled hn encoded hcache messageLift messageBound
    inputBounded hb hgain

end MxxExponentLut
