import GenDecode
import Stage_decode

/-! The looped decode stage, from the generic lemmas of `GenDecode`
(written by `generator/gen_decode_loops.py`). -/

set_option maxRecDepth 65536

namespace DecodeLoops

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof GenDecode

/-! The stage's numbers: 2 components, 8 CRT terms per component, 16 family members,
28 columns, ring dimension 131072, decoding modulus 2135831716638487324604933616719561441822403870837824617769981942184435508314978270190567457292289. -/

/-- The decode stage's inputs. -/
abbrev Inputs := (Fin 16 → Mxx.Primitives.ExactMatrix 2135831716638487324604933616719561441822403870837824617769981942184435508314978270190567457292289 131072 1 28) × (Fin 131072 → Int) × Mxx.Primitives.ExactMatrix 2135831716638487324604933616719561441822403870837824617769981942184435508314978270190567457292289 131072 2 56 × Mxx.Primitives.ExactMatrix 2135831716638487324604933616719561441822403870837824617769981942184435508314978270190567457292289 131072 1 1 × Int × Int × Unit

/-- The factor `r` of component `j`'s expected row. -/
noncomputable def decodeR (inputs : Inputs) : Nat → ExactMatrix 2135831716638487324604933616719561441822403870837824617769981942184435508314978270190567457292289 131072 1 1
  | 0 => inputs.2.2.2.1
  | 1 => (1 : ExactMatrix 2135831716638487324604933616719561441822403870837824617769981942184435508314978270190567457292289 131072 1 1)
  | _ => 0

/-- The value whose CRT terms component `j` reads: its expected row plus the error row `E j`. -/
noncomputable def decodeValue (inputs : Inputs) (G : ExactMatrix 1942566534251559791942281995946931586214205534210157066971588917963081260105189818369 131072 1 28)
    (E : Nat → ErrorMatrix 131072 1 28) (j : Nat) : ExactMatrix 2135831716638487324604933616719561441822403870837824617769981942184435508314978270190567457292289 131072 1 28 :=
  decodeExpected (colBlock 28 inputs.2.2.1 (28 * j))
    (fun _ _ ↦ polynomialOfCoefficients inputs.2.1) 1099489607681 (decodeR inputs j)
    inputs.2.2.2.2.1 inputs.2.2.2.2.2.1 G + reduceMatrix 2135831716638487324604933616719561441822403870837824617769981942184435508314978270190567457292289 131072 1 28 (E j)

theorem stride_step_sequential_generatedRoot_4 {a0 a1} {i : Nat} {cur next} {fam}
    (h : Stage_decode.sequential_generatedRoot_4 a0 a1 i (cur, (fam, ())) next) :
    StrideStep fam 2 0 i cur next := by
  obtain ⟨x, -, -, hx, rfl⟩ := h
  exact ⟨x, hx, rfl⟩

theorem stride_step_sequential_generatedRoot_32 {a0 a1} {i : Nat} {cur next} {fam}
    (h : Stage_decode.sequential_generatedRoot_32 a0 a1 i (cur, (fam, ())) next) :
    StrideStep fam 2 1 i cur next := by
  obtain ⟨x, -, -, hx, rfl⟩ := h
  exact ⟨x, hx, rfl⟩

theorem round_step_sequential_sequential_generatedRoot_28_6 {a0 a1 a2} {k : Nat} {c c' : Int} {vals half qc}
    (h : Stage_decode.sequential_sequential_generatedRoot_28_6 a0 a1 a2 k (c, (vals, (half, (qc, ())))) c') :
    RoundStep vals half qc k c c' := by
  obtain ⟨v, -, -, hv, -, -, rfl⟩ := h
  exact ⟨v, hv, rfl⟩

theorem column_step_sequential_generatedRoot_28 {a0 a1} {i : Nat} {cur next : Int} {D half qc}
    (h : Stage_decode.sequential_generatedRoot_28 a0 a1 i (cur, (D, (half, (qc, ())))) next) :
    ColumnStep D half qc i cur next := by
  obtain ⟨w, -, -, -, -, -, -, hs, hv, -, hrun, rfl⟩ := h
  refine ⟨_, _, _, _, hs, hv, ?_, hrun⟩
  exact fun _ _ _ hb ↦ round_step_sequential_sequential_generatedRoot_28_6 hb

theorem round_step_sequential_sequential_generatedRoot_46_6 {a0 a1 a2} {k : Nat} {c c' : Int} {vals half qc}
    (h : Stage_decode.sequential_sequential_generatedRoot_46_6 a0 a1 a2 k (c, (vals, (half, (qc, ())))) c') :
    RoundStep vals half qc k c c' := by
  obtain ⟨v, -, -, hv, -, -, rfl⟩ := h
  exact ⟨v, hv, rfl⟩

theorem column_step_sequential_generatedRoot_46 {a0 a1} {i : Nat} {cur next : Int} {D half qc}
    (h : Stage_decode.sequential_generatedRoot_46 a0 a1 i (cur, (D, (half, (qc, ())))) next) :
    ColumnStep D half qc i cur next := by
  obtain ⟨w, -, -, -, -, -, -, hs, hv, -, hrun, rfl⟩ := h
  refine ⟨_, _, _, _, hs, hv, ?_, hrun⟩
  exact fun _ _ _ hb ↦ round_step_sequential_sequential_generatedRoot_46_6 hb

/-- The decode stage outputs zero when every CRT term `T (t 2 + j)` it reads is the term of
`y_j + e'_j` for the prime `qN t` and the basis element `KN t`, for the expected row `y_j` of
component `j` and a small error row `e'_j`. -/
theorem decode_zero_loops_gen (qN : Nat → Nat) (KN : Nat → Int) (hq : ∀ i : Fin 8, 1 < qN i)
    (hK : ∀ i j : Fin 8, KN i ≡ (if i = j then 1 else 0) [ZMOD (qN j : Int)])
    (hcop : ∀ i j : Fin 8, i ≠ j → Nat.Coprime (qN i) (qN j)) (hprod : 2135831716638487324604933616719561441822403870837824617769981942184435508314978270190567457292289 = ∏ i : Fin 8, qN i)
    {backend : MxxRuntime.BackendContext} {params : Stage_decode.Params}
    {inputs : Inputs} {out : Int}
    (h : Stage_decode.generatedRoot backend params inputs out)
    {G : ExactMatrix 1942566534251559791942281995946931586214205534210157066971588917963081260105189818369 131072 1 28} (hG : gadgetMatrixRuns backend 1024 28 G)
    (E : Nat → ErrorMatrix 131072 1 28)
    (hE : ∀ j c i, 4 * |(E j 0 c).coeff i| + 4 < (2135831716638487324604933616719561441822403870837824617769981942184435508314978270190567457292289 : Int))
    (hterm : ∀ i : Fin 16, TermOf (qN (i / 2)) (KN (i / 2)) (inputs.1 i)
      (decodeValue inputs G E (i % 2))) :
    out = 0 := by
  obtain ⟨w, ⟨hinit0, -, hs0, hσ, -, -, -, -, -, -, hA0, -, -, -, -, -, -, hr00, -, -, -, -, -, -, hr01, hgad, hreb, -, hc0, hinit1, -, hs1⟩, ⟨-, -, -, -, -, -, hA1, -, -, -, -, -, -, hr10, -, -, -, -, -, -, hr11, -, hc1, hout⟩⟩ := h
  have hσ' := sigma_of_values hσ
  have hg := hreb.2
  rw [gadgetMatrixRuns_unique hgad hG] at hg
  have z0 := decode_component_zero (K := 8) (Rn := 2) (jn := 0) (by decide) (by decide)
    qN KN hq hK hcop hprod (by decide) (by decide)
    inputs.1 (decodeValue inputs G E) hterm _ (E 0) rfl (hE 0) rfl rfl rfl rfl hinit0
    (fun _ _ _ hb ↦ stride_step_sequential_generatedRoot_4 hb) hs0 ?hD0 (by decide) rfl
    (fun _ _ _ hb ↦ column_step_sequential_generatedRoot_28 hb) hc0
  case hD0 =>
    rw [slice_row_gen hr00 rfl (by decide), slice_row_gen hr01 rfl (by decide), slice_colBlock hA0, hσ', hg]
    rfl
  have z1 := decode_component_zero (K := 8) (Rn := 2) (jn := 1) (by decide) (by decide)
    qN KN hq hK hcop hprod (by decide) (by decide)
    inputs.1 (decodeValue inputs G E) hterm _ (E 1) rfl (hE 1) rfl rfl rfl rfl hinit1
    (fun _ _ _ hb ↦ stride_step_sequential_generatedRoot_32 hb) hs1 ?hD1 (by decide) rfl
    (fun _ _ _ hb ↦ column_step_sequential_generatedRoot_46 hb) hc1
  case hD1 =>
    rw [slice_row_gen hr10 rfl (by decide), slice_row_gen hr11 rfl (by decide), slice_colBlock hA1, hσ', hg]
    rfl
  rw [hout, z0, z1]
  try rfl

/-- The CRT primes, read from the online stage's instance table `mod_1` (instance `t 2 + j`). -/
abbrev decode_qN (t : Nat) : Nat := [1099510054913, 1099507695617, 1099504549889, 1099502714881, 1099500617729, 1099499569153, 1099498258433, 1099489607681].getD t 0
/-- The CRT basis, read from the online stage's instance table `sel_12`. -/
abbrev decode_KN (t : Nat) : Int := [1374678475896488657084192654818126458578489200575183305715720471066826246732420366207298579854162, 608845100957124604466115757180559524591508342641850902869547426274875226073018628588519038037007, 285570776699155647957352753009711663432621880987230828076103097971976918194286575494541212286238, 1587294474061516256045012854372943818705185313663117837620253227117089310150346873710447679285576, 1246287434441293117480021120013245985650032798462991164040548808475266533328686994168929822981193, 1018459864462330065985974371762073955448796949294894488143102034870192552613708812836029951210807, 1794116289959797369991619871842198466687482511480982987048347641124756666878163997582030651307155, 628074450076243579409445083879385894195498486245046957566305061836758579289280832174472894207019].getD t 0

theorem decode_gt : ∀ i : Fin 8, 1 < decode_qN i := by decide
theorem decode_basis : ∀ i j : Fin 8, decode_KN i ≡ (if i = j then 1 else 0) [ZMOD (decode_qN j : Int)] := by decide
theorem decode_coprime : ∀ i j : Fin 8, i ≠ j → Nat.Coprime (decode_qN i) (decode_qN j) := by decide
theorem decode_prod : 2135831716638487324604933616719561441822403870837824617769981942184435508314978270190567457292289 = ∏ i : Fin 8, decode_qN i := by
  simp only [Fin.prod_univ_succ, Fin.prod_univ_zero]
  decide

/-- `decode_zero_loops_gen` for the online stage's CRT tables. -/
theorem decode_zero_loops {backend : MxxRuntime.BackendContext} {params : Stage_decode.Params}
    {inputs : Inputs} {out : Int}
    (h : Stage_decode.generatedRoot backend params inputs out)
    {G : ExactMatrix 1942566534251559791942281995946931586214205534210157066971588917963081260105189818369 131072 1 28} (hG : gadgetMatrixRuns backend 1024 28 G)
    (E : Nat → ErrorMatrix 131072 1 28)
    (hE : ∀ j c i, 4 * |(E j 0 c).coeff i| + 4 < (2135831716638487324604933616719561441822403870837824617769981942184435508314978270190567457292289 : Int))
    (hterm : ∀ i : Fin 16, TermOf (decode_qN (i / 2)) (decode_KN (i / 2)) (inputs.1 i)
      (decodeValue inputs G E (i % 2))) :
    out = 0 :=
  decode_zero_loops_gen decode_qN decode_KN decode_gt decode_basis decode_coprime decode_prod h hG E hE hterm

end DecodeLoops
