import Dio2Online0PartPrimes

/-!
# The row of one refresh part of step 0

The five CRT terms of a refresh part (`Dio2Online0PartPrimes`) are terms of the same row when the
primes' rebased gadgets agree (the gadget of `Q_0` is unique); their sum, the part's output
`w_288`, is that row (`part_row`).
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof GenDecode W_Stage_online_0

/-- The primes of `Q_0`. -/
def primes0 : Fin 5 → Nat := ![modulus_1, modulus_2, modulus_3, modulus_4, modulus_5]

/-- The CRT basis of `Q_0` (the refresh parts' scalars). -/
def crt0 : Fin 5 → Int :=
  ![1224430409422355437355602951403039070751149016291967033667224370016023747552,
    1577331461107503857687361648278377550964861539914994428956938744873164216254,
    231620637999516441280060882203998849893024391639068770271694775176939662912,
    286408645742909803465705372805876867840931182364166978575391001671831167302,
    298711634382445948711902742905408453456759239282082071569621589367657095967]

theorem primes0_gt : ∀ i, 1 < primes0 i := by
  intro i; fin_cases i <;> simp [primes0] <;> norm_num [modulus_1, modulus_2, modulus_3, modulus_4, modulus_5]

theorem crt0_basis : ∀ i j, crt0 i ≡ (if i = j then 1 else 0) [ZMOD (primes0 j : Int)] := by
  intro i j
  fin_cases i <;> fin_cases j <;> decide

theorem primes0_coprime : ∀ i j, i ≠ j → Nat.Coprime (primes0 i) (primes0 j) := by
  intro i j h
  fin_cases i <;> fin_cases j <;> first | exact absurd rfl h | decide

theorem primes0_prod : modulus_0 = ∏ i, primes0 i := by
  rw [Fin.prod_univ_five]
  decide

/-- **The part's row**: the sum of the five CRT terms of `v` is `v`. -/
theorem online0_part_row {backend : BackendContext} {hashModel : HashModel} {params : Params}
    {i : Nat} (ctx : parallel_generatedRoot_1281.Ctx backend hashModel params i)
    {v : ExactMatrix modulus_0 8 1 25}
    (h0 : TermOf modulus_1 (crt0 0) (parallel_generatedRoot_1281.w_64_0 ctx) v)
    (h1 : TermOf modulus_2 (crt0 1) (parallel_generatedRoot_1281.w_119_0 ctx) v)
    (h2 : TermOf modulus_3 (crt0 2) (parallel_generatedRoot_1281.w_175_0 ctx) v)
    (h3 : TermOf modulus_4 (crt0 3) (parallel_generatedRoot_1281.w_231_0 ctx) v)
    (h4 : TermOf modulus_5 (crt0 4) (parallel_generatedRoot_1281.w_287_0 ctx) v) :
    parallel_generatedRoot_1281.w_288_0 ctx = v := by
  have h := part_row (by norm_num) (by norm_num [modulus_0]) primes0 primes0_gt crt0 crt0_basis
    primes0_coprime primes0_prod
    (t := ![parallel_generatedRoot_1281.w_64_0 ctx, parallel_generatedRoot_1281.w_119_0 ctx,
      parallel_generatedRoot_1281.w_175_0 ctx, parallel_generatedRoot_1281.w_231_0 ctx,
      parallel_generatedRoot_1281.w_287_0 ctx])
    (fun j ↦ by fin_cases j <;> assumption)
  exact h

end Dio2Proof
