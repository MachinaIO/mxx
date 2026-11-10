import TfheBase

/-!
# One run of `evaluationFails` as a function of the tape

`failR messages ω` is the outcome of `evaluationFails` on the tape `ω`: the keys `keygenOut ω`,
the encryptions `encOut` at sites `15` and `18`, then the deterministic NAND evaluation and
decryption. `probTrue_evaluationFails` turns the failure probability into the tape measure of
`{ω | failR messages ω = true}`.
-/
namespace MxxFheTfhe
open MeasureTheory Mxx.Primitives MxxRuntime MxxSecurity GeneratedScheme

abbrev KeygenOutputs := (Fin 720 → ExactMatrix Q N 1 12) × (Fin 720 → ExactMatrix Q N 1 12) ×
  (Fin 5898240 → Int) × (Fin 8192 → Int) × ExactMatrix Q N 1 1 × ExactMatrix Q N 1 1 ×
  (Fin 5898240 → Int) × (Fin 8192 → Int) × (Fin 720 → Int) × Unit

/-- The keys of a tape. -/
noncomputable def keygenOut (ω : Key → ℤ) : KeygenOutputs :=
  determined fun outputs ↦ ∃ witness : Keygen.generatedRoot.Witness,
    witness.w_0_0 = smp (.residue Q) 0 ω ∧ witness.w_1_0 = smp bitLaw 1 ω ∧
    witness.w_4_0 = smp bitLaw 2 ω ∧ witness.w_5_0 = smp ringLaw 3 ω ∧
    witness.w_7_0 = smp (.residue Qmask) 4 ω ∧ witness.w_21_0 = smp lweLaw 5 ω ∧
    witness.w_23_0 = smp lweLaw 6 ω ∧ witness.w_25_0 = smp lweLaw 7 ω ∧
    witness.w_27_0 = smp lweLaw 8 ω ∧ witness.w_29_0 = smp lweLaw 9 ω ∧
    witness.w_31_0 = smp lweLaw 10 ω ∧ witness.w_33_0 = smp lweLaw 11 ω ∧
    witness.w_35_0 = smp lweLaw 12 ω ∧ witness.w_38_0 = smp (.residue Q) 13 ω ∧
    witness.w_40_0 = smp ringLaw 14 ω ∧
    Keygen.generatedRoot.body Backend.backend Keygen_params () outputs witness

/-- The encryption at sites `base`, `base + 1`, `base + 2`. -/
noncomputable def encOut (base : Nat) (inputs : (Fin 5898240 → Int) × ExactMatrix Q N 1 1 ×
    ExactMatrix Q N 1 1 × Int × (Fin 8192 → Int) × Unit) (ω : Key → ℤ) :
    (Fin 720 → Int) × Int × Unit :=
  determined fun outputs ↦ ∃ witness : Encrypt.generatedRoot.Witness,
    witness.w_1_0 = smp bitLaw base ω ∧ witness.w_4_0 = smp ringLaw (base + 1) ω ∧
    witness.w_23_0 = smp ringLaw (base + 2) ω ∧
    Encrypt.generatedRoot.body Encrypt_params inputs outputs witness

theorem represents_keygen :
    Represents μ (Keygen.generatedRoot.comp Backend.backend Keygen_params ()) (block 0 15)
      keygenOut := by
  obtain ⟨S, hS, h⟩ : ∃ S, S ⊆ block 0 15 ∧
      Represents μ (Keygen.generatedRoot.comp Backend.backend Keygen_params ()) S keygenOut := by
    apply Exists.intro
    refine ⟨?_, ?_⟩
    rotate_left
    apply Represents.congr_fun
    · unfold Keygen.generatedRoot.comp
      simp only [sampleInterval, sampleGaussian, sampleResidue]
      apply represents_smp_bind 0 (law := .residue Q) rfl; intro x0
      apply represents_smp_bind 1 (law := bitLaw) rfl; intro x1
      apply represents_smp_bind 2 (law := bitLaw) rfl; intro x2
      apply represents_smp_bind 3 (law := ringLaw) rfl; intro x3
      apply represents_smp_bind 4 (law := .residue Qmask) rfl; intro x4
      apply represents_smp_bind 5 (law := lweLaw) rfl; intro x5
      apply represents_smp_bind 6 (law := lweLaw) rfl; intro x6
      apply represents_smp_bind 7 (law := lweLaw) rfl; intro x7
      apply represents_smp_bind 8 (law := lweLaw) rfl; intro x8
      apply represents_smp_bind 9 (law := lweLaw) rfl; intro x9
      apply represents_smp_bind 10 (law := lweLaw) rfl; intro x10
      apply represents_smp_bind 11 (law := lweLaw) rfl; intro x11
      apply represents_smp_bind 12 (law := lweLaw) rfl; intro x12
      apply represents_smp_bind 13 (law := .residue Q) rfl; intro x13
      apply represents_smp_bind 14 (law := ringLaw) rfl; intro x14
      exact represents_pure _
      all_goals sites_ne
    · intro ω; rfl
    · sites_sub
  exact h.mono hS

theorem represents_encrypt (base : Nat) (hbase : base = 15 ∨ base = 18) (inputs) :
    Represents μ (Encrypt.generatedRoot.comp Encrypt_params inputs) (block base (base + 3))
      (encOut base inputs) := by
  rcases hbase with rfl | rfl
  · obtain ⟨S, hS, h⟩ : ∃ S, S ⊆ block 15 (15 + 3) ∧
        Represents μ (Encrypt.generatedRoot.comp Encrypt_params inputs) S (encOut 15 inputs) := by
      apply Exists.intro
      refine ⟨?_, ?_⟩
      rotate_left
      apply Represents.congr_fun
      · unfold Encrypt.generatedRoot.comp
        simp only [sampleInterval, sampleGaussian]
        apply represents_smp_bind 15 (law := bitLaw) rfl; intro x
        apply represents_smp_bind 16 (law := ringLaw) rfl; intro y
        apply represents_smp_bind 17 (law := ringLaw) rfl; intro z
        exact represents_pure _
        all_goals sites_ne
      · intro ω; rfl
      · sites_sub
    exact h.mono hS
  · obtain ⟨S, hS, h⟩ : ∃ S, S ⊆ block 18 (18 + 3) ∧
        Represents μ (Encrypt.generatedRoot.comp Encrypt_params inputs) S (encOut 18 inputs) := by
      apply Exists.intro
      refine ⟨?_, ?_⟩
      rotate_left
      apply Represents.congr_fun
      · unfold Encrypt.generatedRoot.comp
        simp only [sampleInterval, sampleGaussian]
        apply represents_smp_bind 18 (law := bitLaw) rfl; intro x
        apply represents_smp_bind 19 (law := ringLaw) rfl; intro y
        apply represents_smp_bind 20 (law := ringLaw) rfl; intro z
        exact represents_pure _
        all_goals sites_ne
      · intro ω; rfl
      · sites_sub
    exact h.mono hS

/-- Whether decryption of the NAND of the encryptions of `messages` fails on the tape `ω`. -/
noncomputable def failR (messages : Fin 2 → Int) (ω : Key → ℤ) : Bool :=
  let x := keygenOut ω
  let x1 := encOut 15 (x.2.2.2.2.2.2.1, x.2.2.2.2.1, x.2.2.2.2.2.1, messages 0,
    x.2.2.2.2.2.2.2.1, ()) ω
  let x2 := encOut 18 (x.2.2.2.2.2.2.1, x.2.2.2.2.1, x.2.2.2.2.2.1, messages 1,
    x.2.2.2.2.2.2.2.1, ()) ω
  let r : (Fin 720 → Int) × Int × Unit := determined fun outputs ↦ ∃ witness,
    Eval.generatedRoot.body Backend.backend Eval_params
      (x.2.2.2.2.2.2.1, x1.2.1, x2.2.1, x1.1, x2.1, x.1, x.2.1, x.2.2.2.2.2.2.2.1, ()) outputs
      witness
  let d : Int := determined fun outputs ↦ ∃ witness,
    Decrypt_1.generatedRoot.body Decrypt_1_params (r.2.1, r.1, x.2.2.2.2.2.2.2.2.1, ()) outputs
      witness
  decide (d ≠ determined fun outputs ↦
    Plain.generatedRoot Plain_params (messages 0, messages 1, ()) outputs)

theorem probTrue_evaluationFails (messages : Fin 2 → Int) :
    probTrue (MxxFhe.Scheme.evaluationFails scheme evaluation messages) =
      Measure.infinitePi μ {ω | failR messages ω = true} := by
  refine Represents.probTrue_eq (S := block 0 15 ∪ (block 15 18 ∪ (block 18 21 ∪ ∅))) ?_
  simp only [MxxFhe.Scheme.evaluationFails, scheme, keyGen, enc, MxxFhe.Scheme.encAll, evaluation,
    bind_assoc, pure_bind, dec, Eval.generatedRoot.comp, Decrypt_1.generatedRoot.comp,
    Fin.zero_eta, Fin.mk_one, Fin.cons_zero, Fin.cons_one, Fin.cons_succ]
  apply Represents.congr_fun
  · apply represents_bind represents_keygen
    · intro x
      apply represents_bind (represents_encrypt 15 (by decide) _)
      · intro x1
        apply represents_bind (represents_encrypt 18 (by decide) _)
        · intro x2
          exact represents_pure _
        · exact Finset.disjoint_empty_right _
      · simp only [Finset.disjoint_union_right, Finset.disjoint_empty_right, and_true]
        exact block_disjoint (by norm_num)
    · simp only [Finset.disjoint_union_right, Finset.disjoint_empty_right, and_true]
      exact ⟨block_disjoint (by norm_num), block_disjoint (by norm_num)⟩
  · intro ω; rfl

end MxxFheTfhe
