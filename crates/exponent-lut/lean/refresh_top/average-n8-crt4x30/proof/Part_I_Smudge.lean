import TopLut
import TopPlace
import TopInst

/-! The smudge LUTs of the decoders (written by `generator/gen_online_luts.py`). -/

namespace RefreshTopProof.Part_I

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

/-- The readout table of a decoder's smudge PRF: the coefficients `2 f(k) - 1`. -/
noncomputable def smudgeTable (k : Fin 4) : ErrorPoly N := ((![1, 1, 3, 3] k : Int) : ErrorPoly N)

/-- The digit weights of the smudge LUT `560`. -/
def smudgeWeight560 (i0 : Nat) : Fin 20 → Int := ![(halfZ (i0 : Int)), 2, 8, 32, 128, 512, 2048, 8192, 32768, 131072, 524288, 2097152, 8388608, 33554432, 134217728, 536870912, 2147483648, 8589934592, 34359738368, 137438953472]

/-- The error bound added by one branch of the LUT `560`. -/
abbrev lutBranchBound560 (i0 : Nat) (B Bh : Nat) : Nat :=
  (6 + 6) * N * Bh * 16384 + (6 + 6) * N * B * 16384 + (N * 1 * ((qd i0) / 2) + (qd i0) / 2)

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- One iteration of the LUT loop `560` adds branch `i` to the accumulated relation and carries
the switching ciphertext of branch `i + 1`. -/
theorem lutStep560 {i0 : Nat} {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    [Fact (i0 < 8)] (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q0 N 1 6} {A : Fin 2 → Fin 2 → ExactMatrix Q0 N 1 6} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix (R i0) N 1 6}
    {i : Nat} (hi : i < 3) {state next : LoopState Q0 (R i0) 6}
    (hswitch : state.switch = fun k j ↦ fam k j ⟨i, by omega⟩)
    {T : Fin 2 → ExactMatrix Q0 N 1 6} {Bs : Nat} (hstate : Rel sm state.key T state.row Bs)
    {params} {o : Nat}
    (hbody : W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_562_18 Backend.backend model params i0 o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0, c 1, A 0 1, A 1 1, fam 0 0,
        fam 0 1, fam 1 0, fam 1 1, ()) next) :
    Rel sm next.key (T + branchTarget (q := Q0) (p := (qd i0)) μ sp fam ⟨i, by omega⟩) next.row
        (Bs + lutBranchBound560 i0 B Bh) ∧
      next.switch = fun k j ↦ fam k j ⟨i + 1, by omega⟩ := by
  obtain ⟨L, hL, hLx, -, hLd⟩ := layoutR i0 hi0
  unfold W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_562_18 at hbody
  extract_lets -merge _params _backend w_0_0 w_48_0 w_67_0 w_70_0 w_73_0 w_76_0 w_30_0 w_49_0 w_31_0 w_50_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_24_0 w_26_0 w_79_0 w_83_0 w_85_0 w_87_0 w_2_0 w_5_0 w_80_0 w_81_0 at hbody
  obtain ⟨w_3_0, hbody⟩ := hbody
  obtain ⟨w_6_0, hbody⟩ := hbody
  obtain ⟨w_7_0, hbody⟩ := hbody
  obtain ⟨w_8_0, hbody⟩ := hbody
  obtain ⟨w_8_concat_1, hbody⟩ := hbody
  obtain ⟨w_10_0, hbody⟩ := hbody
  obtain ⟨w_11_0, hbody⟩ := hbody
  obtain ⟨w_12_0, hbody⟩ := hbody
  obtain ⟨w_13_0, hbody⟩ := hbody
  obtain ⟨w_13_concat_1, hbody⟩ := hbody
  obtain ⟨w_14_0, hbody⟩ := hbody
  obtain ⟨w_16_0, hbody⟩ := hbody
  obtain ⟨w_18_0, hbody⟩ := hbody
  obtain ⟨w_20_0, hbody⟩ := hbody
  obtain ⟨w_21_0, hbody⟩ := hbody
  obtain ⟨w_21_concat_1, hbody⟩ := hbody
  obtain ⟨w_23_0, hbody⟩ := hbody
  obtain ⟨w_25_0, hbody⟩ := hbody
  obtain ⟨w_27_0, hbody⟩ := hbody
  obtain ⟨w_28_0, hbody⟩ := hbody
  obtain ⟨w_28_concat_1, hbody⟩ := hbody
  obtain ⟨w_29_0, hbody⟩ := hbody
  obtain ⟨w_32_0, hbody⟩ := hbody
  obtain ⟨w_33_0, hbody⟩ := hbody
  obtain ⟨w_34_0, hbody⟩ := hbody
  extract_lets -merge w_35_0 at hbody
  obtain ⟨w_36_0, hbody⟩ := hbody
  obtain ⟨w_37_0, hbody⟩ := hbody
  obtain ⟨w_38_0, hbody⟩ := hbody
  obtain ⟨w_39_0, hbody⟩ := hbody
  obtain ⟨w_40_0, hbody⟩ := hbody
  obtain ⟨w_41_0, hbody⟩ := hbody
  obtain ⟨w_42_0, hbody⟩ := hbody
  extract_lets -merge w_43_0 at hbody
  obtain ⟨w_44_0, hbody⟩ := hbody
  obtain ⟨w_45_0, hbody⟩ := hbody
  extract_lets -merge w_46_0 w_47_0 at hbody
  obtain ⟨w_51_0, hbody⟩ := hbody
  obtain ⟨w_52_0, hbody⟩ := hbody
  obtain ⟨w_53_0, hbody⟩ := hbody
  extract_lets -merge w_54_0 at hbody
  obtain ⟨w_55_0, hbody⟩ := hbody
  obtain ⟨w_56_0, hbody⟩ := hbody
  obtain ⟨w_57_0, hbody⟩ := hbody
  obtain ⟨w_58_0, hbody⟩ := hbody
  obtain ⟨w_59_0, hbody⟩ := hbody
  obtain ⟨w_60_0, hbody⟩ := hbody
  obtain ⟨w_61_0, hbody⟩ := hbody
  extract_lets -merge w_62_0 at hbody
  obtain ⟨w_63_0, hbody⟩ := hbody
  obtain ⟨w_64_0, hbody⟩ := hbody
  extract_lets -merge w_65_0 w_66_0 at hbody
  obtain ⟨w_68_0, hbody⟩ := hbody
  extract_lets -merge w_69_0 at hbody
  obtain ⟨w_71_0, hbody⟩ := hbody
  extract_lets -merge w_72_0 at hbody
  obtain ⟨w_74_0, hbody⟩ := hbody
  extract_lets -merge w_75_0 at hbody
  obtain ⟨w_77_0, hbody⟩ := hbody
  extract_lets -merge w_78_0 at hbody
  obtain ⟨w_82_0, hbody⟩ := hbody
  obtain ⟨w_84_0, hbody⟩ := hbody
  obtain ⟨w_86_0, hbody⟩ := hbody
  obtain ⟨w_88_0, hbody⟩ := hbody
  simp only [W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_562_18.constraints_0, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_562_18.constraints_1, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_562_18.constraints_2, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_562_18.constraints_3, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_562_18.constraints_4] at hbody
  obtain ⟨h2_0, h2_0_1, h3_0, -, -, h6_0, h7_0, h6_0_1, h8_0, h2_0_2, h2_0_3, h10_0, h11_0, h12_0, h11_0_1, h13_0, h14_0, -, -, h16_0, -, -, h18_0, -, -, h20_0, h18_0_1, h21_0, -, -, h23_0, -, -, h25_0, -, -, h27_0, h25_0_1, h28_0, h29_0, h32_0, h33_0, h34_0, -, -, -, -, -, -, h36_0, h37_0, -, -, -, -, -, -, h38_0, h39_0, h40_0, h41_0, h42_0, -, -, -, -, -, -, h44_0, -, -, -, -, -, -, h45_0, h51_0, h52_0, h53_0, -, -, -, -, -, -, h55_0, h56_0, -, -, -, -, -, -, h57_0, h58_0, h59_0, h60_0, h61_0, -, -, -, -, -, -, h63_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h77_0, h81_0, h81_0_1, h82_0, h81_0_2, h81_0_3, h84_0, h81_0_4, h81_0_5, h86_0, h81_0_6, h81_0_7, h88_0, hout⟩ := hbody
  have hidx : w_2_0 = Int.ofNat i := rfl
  have hnext : w_81_0 = Int.ofNat (i + 1) := by
    simp only [w_81_0, w_80_0, w_2_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h3_0 h10_0
  rw [hnext] at h82_0 h84_0 h86_0 h88_0
  obtain ⟨_, hw3⟩ := familyGet_eq h3_0
  obtain ⟨_, hw10⟩ := familyGet_eq h10_0
  obtain ⟨_, hs0⟩ := familyGet_eq h82_0
  obtain ⟨_, hs1⟩ := familyGet_eq h84_0
  obtain ⟨_, hs2⟩ := familyGet_eq h86_0
  obtain ⟨_, hs3⟩ := familyGet_eq h88_0
  subst hw3 hw10
  have hhelp' := hhelp ⟨i, by omega⟩
  have hAd : lutA model key Q0 6 2 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  simp only [hAd] at hhelp'
  have hbranch := lut_branch (n := N) (q := Q0) (q' := (R i0)) (p := (qd i0)) (m := 6) (m' := 6)
    (log := 3) (bound := 16384) (index := sigma i) hN hpow (q0_eqR i0 hi0) (by inst_num) (by inst_num)
    L hL hLx hg' hg hsm
    henc h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (bsw := state.switch) (Xb := ![w_32_0, w_51_0])
    (fun j ↦ by fin_cases j; exacts [h32_0, h51_0])
    (D' := ![w_33_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h33_0, h52_0])
    (D := ![w_34_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h34_0, h53_0])
    (M := ![![w_36_0, w_55_0], ![w_38_0, w_57_0]])
    (fun j ↦ by fin_cases j; exacts [h36_0, h55_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h57_0])
    (Asw := ![![w_37_0, w_56_0], ![w_39_0, w_58_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h37_0, h56_0, h39_0, h58_0])
    (Z := ![w_40_0, w_59_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h59_0])
    (E' := ![w_41_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h41_0, h60_0])
    (E := ![w_42_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h42_0, h61_0])
    (h := ![_, _]) (sigma_odd ⟨i, by omega⟩) hhelp' h6_0_1 h8_0 h11_0_1 h13_0 h14_0
    (Y := ![w_44_0, w_63_0]) (X := ![w_45_0, w_64_0])
    (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h44_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h45_0, h64_0])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  subst hout
  rw [hswitch] at hbranch
  refine ⟨?_, ?_⟩
  · convert hstate.add hbranch using 1
    · funext k j
      fin_cases k <;> fin_cases j <;> rfl
    · funext j
      fin_cases j <;> rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hs0, hs1, hs2, hs3]

/-- The generated loop `560` accumulates the first `count` branches and carries the switching
ciphertext of branch `count`. -/
theorem lutLoop560 {i0 : Nat} {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    [Fact (i0 < 8)] (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q0 N 1 6} {A : Fin 2 → Fin 2 → ExactMatrix Q0 N 1 6} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix (R i0) N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q0 (R i0) 6} {params} {o : Nat}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q0 (R i0) 6) ↦
      W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_562_18 Backend.backend model params i0 o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1,
          current.2.2.2.2.2.1, current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1,
          current.2.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0,
          c 1, A 0 1, A 1 1, fam 0 0, fam 0 1, fam 1 0, fam 1 1, ()) next) count
      (0, 0, 0, 0, 0, 0, fam 0 0 0, fam 0 1 0, fam 1 0 0, fam 1 1 0, ()) final) :
    Rel sm final.key (∑ i ∈ Finset.range count, lutBranchTarget (q := Q0) (p := (qd i0)) μ sp fam i)
        final.row (count * lutBranchBound560 i0 B Bh) ∧
      final.switch = fun k j ↦ fam k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q0 (R i0) 6) ↦ ∀ hk : k ≤ 3,
      Rel sm s.key (∑ i ∈ Finset.range k, lutBranchTarget (q := Q0) (p := (qd i0)) μ sp fam i) s.row
        (k * lutBranchBound560 i0 B Bh) ∧
        s.switch = fun k' j ↦ fam k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨?_, ?_⟩
    · rw [Finset.range_zero, Finset.sum_empty, zero_mul]
      have hrow : (![0, 0] : Fin 2 → ExactMatrix Q0 N 1 6) = 0 := by
        funext j
        fin_cases j <;> rfl
      simp only [LoopState.key, LoopState.row, hrow]
      exact Rel.zero
    · funext k j
      fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := lutStep560 hsm hg' hg henc hhelp (by omega) hsw hrel hbody
    refine ⟨?_, hsw'⟩
    rw [Finset.sum_range_succ, lutBranchTarget, dif_pos (by omega), add_mul, one_mul]
    exact hrel'

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem lut560 {i0 : Nat} {model : HashModel} {key : ByteArray} {params} {o : Nat} [Fact (i0 < 8)]
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix (R i0) N 1 6} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6} {outputs : Blocks (R i0) 1}
    (h : W_Stage_online_eval.parallel_parallel_generatedRoot_290_562 Backend.backend model params i0 o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11, ()) outputs)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {z : Nat}
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {B : Nat}
    (hinput : Rel sm ![![A00, A01], ![A10, A11]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * Xe (n := N) (2 * z) * unitPair sp j) • g)
      ![c0, c1] B)
    {Bh : Nat} (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    (ho : o < 160) {gq : ExactMatrix (R i0) N 1 6} (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq) {inverse : Nat}
    (hinverse : (inverse : ExactPoly (R i0) N) * ((4 : Nat) : ExactPoly (R i0) N) = 1) {BB : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient (R i0) inverse (fun k ↦ reducePoly (R i0) N (smudgeTable k)) (sigma i).toNat *
          reducePoly (R i0) N ((1 : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix (R i0) N 1 6 (e j)) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly (R i0) N (Xe (n := N) (o / 20)) • reducePoly (R i0) N
        (((smudgeWeight560 i0 ⟨o % 20, Nat.mod_lt _ (by inst_num)⟩ : Int) : ErrorPoly N) *
          ((1 : ErrorPoly N) * ((((-1 : Int) ^ (z / 4) : Int) : ErrorPoly N) *
            smudgeTable ⟨z % 4, Nat.mod_lt _ (by inst_num)⟩)) * unitPair sp j) •
        (1 : ExactMatrix (R i0) N 1 1))
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      (6 * ((3 * lutBranchBound560 i0 B Bh + ((6 + 6) * N * Bh * 16384 + (6 + 6) * N * B * 16384 +
        (N * 1 * ((qd i0) / 2) + (qd i0) / 2)) + (N + 2) * ((qd i0) / 2)) / (qd i0) + 4 * (N * 1 * BB)) * (32768 / 2)) := by
  obtain ⟨L, hL, hLx, -, hLd⟩ := layoutR i0 hi0
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_562.body at h
  extract_lets -merge _params _backend w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_73_concat_1 w_73_concat_2 w_73_concat_3 w_73_concat_4 w_73_concat_5 w_73_concat_6 w_73_concat_7 w_73_concat_8 w_73_concat_9 w_73_concat_10 w_73_concat_11 w_73_concat_12 w_73_concat_13 w_73_concat_14 w_73_concat_15 w_73_concat_16 w_73_concat_17 w_73_concat_18 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 at h
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_562.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_562.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_562.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_562.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_562.constraints_4, W_Stage_online_eval.parallel_parallel_generatedRoot_290_562.constraints_5] at h
  obtain ⟨h2_0, h4_0, h6_0, h8_0, -, h18_9, h19_0, -, -, h21_0, h22_0, h21_0_1, h23_0, h24_0, h25_0, h26_0, h25_0_1, h27_0, h28_0, -, -, h29_0, -, -, h30_0, -, -, h31_0, h30_0_1, h32_0, -, -, h33_0, -, -, h34_0, -, -, h35_0, h34_0_1, h36_0, h37_0, h38_0, h39_0, h40_0, -, -, -, -, -, -, h42_0, h43_0, -, -, -, -, -, -, h44_0, h45_0, h46_0, h47_0, h48_0, -, -, -, -, -, -, h50_0, h52_0, -, -, h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, -, -, -, -, -, -, -, -, h74_0, h75_0, h76_0, -, h80_0, h81_0, h82_0, -, -, -, -, -, -, h84_0, h85_0, -, -, -, -, -, -, h86_0, h87_0, h88_0, h89_0, h90_0, -, -, -, -, -, -, h92_0, h94_0, -, -, -, -, -, -, h97_0, h99_0, -, -, -, -, -, -, h102_0, h104_0, -, -, -, -, -, -, h107_0, -, -, -, -, -, -, h108_0, h111_0, -, -, -, -, -, -, h114_0, -, -, -, -, -, -, h115_0, h118_0, hout⟩ := h
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  rw [hwa, hwb, hwc, hwd] at h18_9
  rw [show Int.toNat 3 = 3 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := lutLoop560 (fam := fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j)
    (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]])
    (final := (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) hsm hg' hg hinput hhelp (count := 3) (by decide) h18_9
  obtain ⟨_, hw19⟩ := familyGet_eq (k := 3) h19_0
  obtain ⟨_, hw24⟩ := familyGet_eq (k := 3) h24_0
  rw [hw19] at h21_0_1
  rw [hw24] at h25_0_1
  have hA0 : w_21_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h21_0
  have hA1 : w_25_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h25_0
  have hA2 : w_22_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h22_0
  have hA3 : w_26_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h26_0
  rw [hA0] at h21_0_1
  rw [hA2] at h23_0
  rw [hA1] at h25_0_1
  rw [hA3] at h27_0
  have hhelp3 := hhelp 3
  have hfin := lut_round (q := Q0) (q' := (R i0)) (p := (qd i0)) (m := 6) (m' := 6) (q0_eqR i0 hi0) (by inst_num)
    (by inst_num) (by inst_num) L hL hLx hsm hg' hg hinput hloop hloopSwitch hhelp3
    h29_0 h30_0 h31_0 h33_0 h34_0 h35_0 h30_0_1 h32_0 h34_0_1 h36_0 h37_0
    (Xb := ![w_38_0, w_80_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h80_0])
    (D' := ![w_39_0, w_81_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h81_0])
    (D := ![w_40_0, w_82_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h82_0])
    (M := ![![w_42_0, w_84_0], ![w_44_0, w_86_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h84_0])
    (fun j ↦ by fin_cases j; exacts [h44_0, h86_0])
    (Asw := ![![w_43_0, w_85_0], ![w_45_0, w_87_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h43_0, h85_0, h45_0, h87_0])
    (Z := ![w_46_0, w_88_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h88_0])
    (E' := ![w_47_0, w_89_0]) (fun j ↦ by fin_cases j; exacts [h47_0, h89_0])
    (E := ![w_48_0, w_90_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h90_0])
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (Y := ![w_107_0, w_114_0]) (X := ![w_108_0, w_115_0])
    (K := ![![w_50_0, w_92_0], ![w_97_0, w_102_0]])
    (fun j ↦ by fin_cases j; exacts [h107_0, h114_0])
    (fun j ↦ by fin_cases j; exacts [h108_0, h115_0])
    (fun j ↦ by fin_cases j; exacts [h50_0, h92_0])
    (fun j ↦ by fin_cases j; exacts [h97_0, h102_0])
    (T := smudgeTable) (Δ' := (1 : ErrorPoly N)) hinverse
    (fun i ↦ hswitch i) (K' := ![![w_52_0, w_94_0], ![w_99_0, w_104_0]])
    (R' := ![w_111_0, w_118_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h52_0, h94_0, h99_0, h104_0])
    (fun j ↦ by fin_cases j; exacts [h111_0, h118_0])
  -- The weight `w_(o mod 20)` of the member's digit.
  simp only [concatColumns] at h54_0 h55_0 h56_0 h57_0 h58_0 h59_0 h60_0 h61_0 h62_0 h63_0 h64_0 h65_0 h66_0 h67_0 h68_0 h69_0 h70_0 h71_0 h73_0
  have hrow : ∀ c : Fin 20, w_73_0 0 c = (matrixPolynomial [smudgeWeight560 i0 c] : ExactMatrix (R i0) N 1 1) 0 0 := by
    intro c
    fin_cases c <;> simp [h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, w_53_0, w_54_0, w_55_0, w_56_0, w_57_0, w_58_0, w_59_0, w_60_0, w_61_0, w_62_0, w_63_0, w_64_0, w_65_0, w_66_0, w_67_0, w_68_0, w_69_0, w_70_0, w_71_0, w_72_0, smudgeWeight560]
  have hd : Int.fmod (Int.ofNat o) 20 = ((o % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by inst_num)]; rfl
  rw [hd] at h74_0
  have hK : w_74_0 = matrixPolynomial [smudgeWeight560 i0 ⟨o % 20, Nat.mod_lt _ (by inst_num)⟩] := by
    funext a b
    obtain rfl : a = 0 := Subsingleton.elim _ _
    obtain rfl : b = 0 := Subsingleton.elim _ _
    rw [slice_entry h74_0 (by simp; omega)]
    exact hrow _
  have hfin2 := hfin.finalizeCoefficient (by inst_num) hN L hL hLx hgq _ hK h75_0 h76_0
  have hpl := rel_monomial hfin2 (k := o / 20) (by simp only [N]; omega)
  have hq : w_78_0 = ((o / 20 : Nat) : Int) := by
    simp only [w_78_0]
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by inst_num)]; rfl
  rw [hout]
  convert hpl using 1
  all_goals simp only [w_79_0, w_96_0, w_101_0, w_106_0, w_113_0, w_120_0, w_77_0, w_95_0, w_100_0, w_105_0, w_112_0, w_119_0, hq]
  all_goals first
    | (funext a j; fin_cases a <;> fin_cases j <;> rfl)
    | (funext j; fin_cases j <;> rfl)

/-- The digit weights of the smudge LUT `615`. -/
def smudgeWeight615 (i0 : Nat) : Fin 20 → Int := ![(halfZ (i0 : Int)), 2, 8, 32, 128, 512, 2048, 8192, 32768, 131072, 524288, 2097152, 8388608, 33554432, 134217728, 536870912, 2147483648, 8589934592, 34359738368, 137438953472]

/-- The error bound added by one branch of the LUT `615`. -/
abbrev lutBranchBound615 (i0 : Nat) (B Bh : Nat) : Nat :=
  (6 + 6) * N * Bh * 16384 + (6 + 6) * N * B * 16384 + (N * 1 * ((qd i0) / 2) + (qd i0) / 2)

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- One iteration of the LUT loop `615` adds branch `i` to the accumulated relation and carries
the switching ciphertext of branch `i + 1`. -/
theorem lutStep615 {i0 : Nat} {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    [Fact (i0 < 8)] (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q0 N 1 6} {A : Fin 2 → Fin 2 → ExactMatrix Q0 N 1 6} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix (R i0) N 1 6}
    {i : Nat} (hi : i < 3) {state next : LoopState Q0 (R i0) 6}
    (hswitch : state.switch = fun k j ↦ fam k j ⟨i, by omega⟩)
    {T : Fin 2 → ExactMatrix Q0 N 1 6} {Bs : Nat} (hstate : Rel sm state.key T state.row Bs)
    {params} {o : Nat}
    (hbody : W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_617_18 Backend.backend model params i0 o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0, c 1, A 0 1, A 1 1, fam 0 0,
        fam 0 1, fam 1 0, fam 1 1, ()) next) :
    Rel sm next.key (T + branchTarget (q := Q0) (p := (qd i0)) μ sp fam ⟨i, by omega⟩) next.row
        (Bs + lutBranchBound615 i0 B Bh) ∧
      next.switch = fun k j ↦ fam k j ⟨i + 1, by omega⟩ := by
  obtain ⟨L, hL, hLx, -, hLd⟩ := layoutR i0 hi0
  unfold W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_617_18 at hbody
  extract_lets -merge _params _backend w_0_0 w_48_0 w_67_0 w_70_0 w_73_0 w_76_0 w_30_0 w_49_0 w_31_0 w_50_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_24_0 w_26_0 w_79_0 w_83_0 w_85_0 w_87_0 w_2_0 w_5_0 w_80_0 w_81_0 at hbody
  obtain ⟨w_3_0, hbody⟩ := hbody
  obtain ⟨w_6_0, hbody⟩ := hbody
  obtain ⟨w_7_0, hbody⟩ := hbody
  obtain ⟨w_8_0, hbody⟩ := hbody
  obtain ⟨w_8_concat_1, hbody⟩ := hbody
  obtain ⟨w_10_0, hbody⟩ := hbody
  obtain ⟨w_11_0, hbody⟩ := hbody
  obtain ⟨w_12_0, hbody⟩ := hbody
  obtain ⟨w_13_0, hbody⟩ := hbody
  obtain ⟨w_13_concat_1, hbody⟩ := hbody
  obtain ⟨w_14_0, hbody⟩ := hbody
  obtain ⟨w_16_0, hbody⟩ := hbody
  obtain ⟨w_18_0, hbody⟩ := hbody
  obtain ⟨w_20_0, hbody⟩ := hbody
  obtain ⟨w_21_0, hbody⟩ := hbody
  obtain ⟨w_21_concat_1, hbody⟩ := hbody
  obtain ⟨w_23_0, hbody⟩ := hbody
  obtain ⟨w_25_0, hbody⟩ := hbody
  obtain ⟨w_27_0, hbody⟩ := hbody
  obtain ⟨w_28_0, hbody⟩ := hbody
  obtain ⟨w_28_concat_1, hbody⟩ := hbody
  obtain ⟨w_29_0, hbody⟩ := hbody
  obtain ⟨w_32_0, hbody⟩ := hbody
  obtain ⟨w_33_0, hbody⟩ := hbody
  obtain ⟨w_34_0, hbody⟩ := hbody
  extract_lets -merge w_35_0 at hbody
  obtain ⟨w_36_0, hbody⟩ := hbody
  obtain ⟨w_37_0, hbody⟩ := hbody
  obtain ⟨w_38_0, hbody⟩ := hbody
  obtain ⟨w_39_0, hbody⟩ := hbody
  obtain ⟨w_40_0, hbody⟩ := hbody
  obtain ⟨w_41_0, hbody⟩ := hbody
  obtain ⟨w_42_0, hbody⟩ := hbody
  extract_lets -merge w_43_0 at hbody
  obtain ⟨w_44_0, hbody⟩ := hbody
  obtain ⟨w_45_0, hbody⟩ := hbody
  extract_lets -merge w_46_0 w_47_0 at hbody
  obtain ⟨w_51_0, hbody⟩ := hbody
  obtain ⟨w_52_0, hbody⟩ := hbody
  obtain ⟨w_53_0, hbody⟩ := hbody
  extract_lets -merge w_54_0 at hbody
  obtain ⟨w_55_0, hbody⟩ := hbody
  obtain ⟨w_56_0, hbody⟩ := hbody
  obtain ⟨w_57_0, hbody⟩ := hbody
  obtain ⟨w_58_0, hbody⟩ := hbody
  obtain ⟨w_59_0, hbody⟩ := hbody
  obtain ⟨w_60_0, hbody⟩ := hbody
  obtain ⟨w_61_0, hbody⟩ := hbody
  extract_lets -merge w_62_0 at hbody
  obtain ⟨w_63_0, hbody⟩ := hbody
  obtain ⟨w_64_0, hbody⟩ := hbody
  extract_lets -merge w_65_0 w_66_0 at hbody
  obtain ⟨w_68_0, hbody⟩ := hbody
  extract_lets -merge w_69_0 at hbody
  obtain ⟨w_71_0, hbody⟩ := hbody
  extract_lets -merge w_72_0 at hbody
  obtain ⟨w_74_0, hbody⟩ := hbody
  extract_lets -merge w_75_0 at hbody
  obtain ⟨w_77_0, hbody⟩ := hbody
  extract_lets -merge w_78_0 at hbody
  obtain ⟨w_82_0, hbody⟩ := hbody
  obtain ⟨w_84_0, hbody⟩ := hbody
  obtain ⟨w_86_0, hbody⟩ := hbody
  obtain ⟨w_88_0, hbody⟩ := hbody
  simp only [W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_617_18.constraints_0, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_617_18.constraints_1, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_617_18.constraints_2, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_617_18.constraints_3, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_617_18.constraints_4] at hbody
  obtain ⟨h2_0, h2_0_1, h3_0, -, -, h6_0, h7_0, h6_0_1, h8_0, h2_0_2, h2_0_3, h10_0, h11_0, h12_0, h11_0_1, h13_0, h14_0, -, -, h16_0, -, -, h18_0, -, -, h20_0, h18_0_1, h21_0, -, -, h23_0, -, -, h25_0, -, -, h27_0, h25_0_1, h28_0, h29_0, h32_0, h33_0, h34_0, -, -, -, -, -, -, h36_0, h37_0, -, -, -, -, -, -, h38_0, h39_0, h40_0, h41_0, h42_0, -, -, -, -, -, -, h44_0, -, -, -, -, -, -, h45_0, h51_0, h52_0, h53_0, -, -, -, -, -, -, h55_0, h56_0, -, -, -, -, -, -, h57_0, h58_0, h59_0, h60_0, h61_0, -, -, -, -, -, -, h63_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h77_0, h81_0, h81_0_1, h82_0, h81_0_2, h81_0_3, h84_0, h81_0_4, h81_0_5, h86_0, h81_0_6, h81_0_7, h88_0, hout⟩ := hbody
  have hidx : w_2_0 = Int.ofNat i := rfl
  have hnext : w_81_0 = Int.ofNat (i + 1) := by
    simp only [w_81_0, w_80_0, w_2_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h3_0 h10_0
  rw [hnext] at h82_0 h84_0 h86_0 h88_0
  obtain ⟨_, hw3⟩ := familyGet_eq h3_0
  obtain ⟨_, hw10⟩ := familyGet_eq h10_0
  obtain ⟨_, hs0⟩ := familyGet_eq h82_0
  obtain ⟨_, hs1⟩ := familyGet_eq h84_0
  obtain ⟨_, hs2⟩ := familyGet_eq h86_0
  obtain ⟨_, hs3⟩ := familyGet_eq h88_0
  subst hw3 hw10
  have hhelp' := hhelp ⟨i, by omega⟩
  have hAd : lutA model key Q0 6 2 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  simp only [hAd] at hhelp'
  have hbranch := lut_branch (n := N) (q := Q0) (q' := (R i0)) (p := (qd i0)) (m := 6) (m' := 6)
    (log := 3) (bound := 16384) (index := sigma i) hN hpow (q0_eqR i0 hi0) (by inst_num) (by inst_num)
    L hL hLx hg' hg hsm
    henc h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (bsw := state.switch) (Xb := ![w_32_0, w_51_0])
    (fun j ↦ by fin_cases j; exacts [h32_0, h51_0])
    (D' := ![w_33_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h33_0, h52_0])
    (D := ![w_34_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h34_0, h53_0])
    (M := ![![w_36_0, w_55_0], ![w_38_0, w_57_0]])
    (fun j ↦ by fin_cases j; exacts [h36_0, h55_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h57_0])
    (Asw := ![![w_37_0, w_56_0], ![w_39_0, w_58_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h37_0, h56_0, h39_0, h58_0])
    (Z := ![w_40_0, w_59_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h59_0])
    (E' := ![w_41_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h41_0, h60_0])
    (E := ![w_42_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h42_0, h61_0])
    (h := ![_, _]) (sigma_odd ⟨i, by omega⟩) hhelp' h6_0_1 h8_0 h11_0_1 h13_0 h14_0
    (Y := ![w_44_0, w_63_0]) (X := ![w_45_0, w_64_0])
    (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h44_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h45_0, h64_0])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  subst hout
  rw [hswitch] at hbranch
  refine ⟨?_, ?_⟩
  · convert hstate.add hbranch using 1
    · funext k j
      fin_cases k <;> fin_cases j <;> rfl
    · funext j
      fin_cases j <;> rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hs0, hs1, hs2, hs3]

/-- The generated loop `615` accumulates the first `count` branches and carries the switching
ciphertext of branch `count`. -/
theorem lutLoop615 {i0 : Nat} {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    [Fact (i0 < 8)] (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q0 N 1 6} {A : Fin 2 → Fin 2 → ExactMatrix Q0 N 1 6} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix (R i0) N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q0 (R i0) 6} {params} {o : Nat}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q0 (R i0) 6) ↦
      W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_617_18 Backend.backend model params i0 o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1,
          current.2.2.2.2.2.1, current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1,
          current.2.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0,
          c 1, A 0 1, A 1 1, fam 0 0, fam 0 1, fam 1 0, fam 1 1, ()) next) count
      (0, 0, 0, 0, 0, 0, fam 0 0 0, fam 0 1 0, fam 1 0 0, fam 1 1 0, ()) final) :
    Rel sm final.key (∑ i ∈ Finset.range count, lutBranchTarget (q := Q0) (p := (qd i0)) μ sp fam i)
        final.row (count * lutBranchBound615 i0 B Bh) ∧
      final.switch = fun k j ↦ fam k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q0 (R i0) 6) ↦ ∀ hk : k ≤ 3,
      Rel sm s.key (∑ i ∈ Finset.range k, lutBranchTarget (q := Q0) (p := (qd i0)) μ sp fam i) s.row
        (k * lutBranchBound615 i0 B Bh) ∧
        s.switch = fun k' j ↦ fam k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨?_, ?_⟩
    · rw [Finset.range_zero, Finset.sum_empty, zero_mul]
      have hrow : (![0, 0] : Fin 2 → ExactMatrix Q0 N 1 6) = 0 := by
        funext j
        fin_cases j <;> rfl
      simp only [LoopState.key, LoopState.row, hrow]
      exact Rel.zero
    · funext k j
      fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := lutStep615 hsm hg' hg henc hhelp (by omega) hsw hrel hbody
    refine ⟨?_, hsw'⟩
    rw [Finset.sum_range_succ, lutBranchTarget, dif_pos (by omega), add_mul, one_mul]
    exact hrel'

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem lut615 {i0 : Nat} {model : HashModel} {key : ByteArray} {params} {o : Nat} [Fact (i0 < 8)]
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix (R i0) N 1 6} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6} {outputs : Blocks (R i0) 1}
    (h : W_Stage_online_eval.parallel_parallel_generatedRoot_290_617 Backend.backend model params i0 o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11, ()) outputs)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {z : Nat}
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {B : Nat}
    (hinput : Rel sm ![![A00, A01], ![A10, A11]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * Xe (n := N) (2 * z) * unitPair sp j) • g)
      ![c0, c1] B)
    {Bh : Nat} (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    (ho : o < 160) {gq : ExactMatrix (R i0) N 1 6} (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq) {inverse : Nat}
    (hinverse : (inverse : ExactPoly (R i0) N) * ((4 : Nat) : ExactPoly (R i0) N) = 1) {BB : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient (R i0) inverse (fun k ↦ reducePoly (R i0) N (smudgeTable k)) (sigma i).toNat *
          reducePoly (R i0) N ((1 : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix (R i0) N 1 6 (e j)) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly (R i0) N (Xe (n := N) (o / 20)) • reducePoly (R i0) N
        (((smudgeWeight615 i0 ⟨o % 20, Nat.mod_lt _ (by inst_num)⟩ : Int) : ErrorPoly N) *
          ((1 : ErrorPoly N) * ((((-1 : Int) ^ (z / 4) : Int) : ErrorPoly N) *
            smudgeTable ⟨z % 4, Nat.mod_lt _ (by inst_num)⟩)) * unitPair sp j) •
        (1 : ExactMatrix (R i0) N 1 1))
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      (6 * ((3 * lutBranchBound615 i0 B Bh + ((6 + 6) * N * Bh * 16384 + (6 + 6) * N * B * 16384 +
        (N * 1 * ((qd i0) / 2) + (qd i0) / 2)) + (N + 2) * ((qd i0) / 2)) / (qd i0) + 4 * (N * 1 * BB)) * (32768 / 2)) := by
  obtain ⟨L, hL, hLx, -, hLd⟩ := layoutR i0 hi0
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_617.body at h
  extract_lets -merge _params _backend w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_73_concat_1 w_73_concat_2 w_73_concat_3 w_73_concat_4 w_73_concat_5 w_73_concat_6 w_73_concat_7 w_73_concat_8 w_73_concat_9 w_73_concat_10 w_73_concat_11 w_73_concat_12 w_73_concat_13 w_73_concat_14 w_73_concat_15 w_73_concat_16 w_73_concat_17 w_73_concat_18 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 at h
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_617.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_617.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_617.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_617.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_617.constraints_4, W_Stage_online_eval.parallel_parallel_generatedRoot_290_617.constraints_5] at h
  obtain ⟨h2_0, h4_0, h6_0, h8_0, -, h18_9, h19_0, -, -, h21_0, h22_0, h21_0_1, h23_0, h24_0, h25_0, h26_0, h25_0_1, h27_0, h28_0, -, -, h29_0, -, -, h30_0, -, -, h31_0, h30_0_1, h32_0, -, -, h33_0, -, -, h34_0, -, -, h35_0, h34_0_1, h36_0, h37_0, h38_0, h39_0, h40_0, -, -, -, -, -, -, h42_0, h43_0, -, -, -, -, -, -, h44_0, h45_0, h46_0, h47_0, h48_0, -, -, -, -, -, -, h50_0, h52_0, -, -, h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, -, -, -, -, -, -, -, -, h74_0, h75_0, h76_0, -, h80_0, h81_0, h82_0, -, -, -, -, -, -, h84_0, h85_0, -, -, -, -, -, -, h86_0, h87_0, h88_0, h89_0, h90_0, -, -, -, -, -, -, h92_0, h94_0, -, -, -, -, -, -, h97_0, h99_0, -, -, -, -, -, -, h102_0, h104_0, -, -, -, -, -, -, h107_0, -, -, -, -, -, -, h108_0, h111_0, -, -, -, -, -, -, h114_0, -, -, -, -, -, -, h115_0, h118_0, hout⟩ := h
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  rw [hwa, hwb, hwc, hwd] at h18_9
  rw [show Int.toNat 3 = 3 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := lutLoop615 (fam := fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j)
    (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]])
    (final := (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) hsm hg' hg hinput hhelp (count := 3) (by decide) h18_9
  obtain ⟨_, hw19⟩ := familyGet_eq (k := 3) h19_0
  obtain ⟨_, hw24⟩ := familyGet_eq (k := 3) h24_0
  rw [hw19] at h21_0_1
  rw [hw24] at h25_0_1
  have hA0 : w_21_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h21_0
  have hA1 : w_25_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h25_0
  have hA2 : w_22_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h22_0
  have hA3 : w_26_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h26_0
  rw [hA0] at h21_0_1
  rw [hA2] at h23_0
  rw [hA1] at h25_0_1
  rw [hA3] at h27_0
  have hhelp3 := hhelp 3
  have hfin := lut_round (q := Q0) (q' := (R i0)) (p := (qd i0)) (m := 6) (m' := 6) (q0_eqR i0 hi0) (by inst_num)
    (by inst_num) (by inst_num) L hL hLx hsm hg' hg hinput hloop hloopSwitch hhelp3
    h29_0 h30_0 h31_0 h33_0 h34_0 h35_0 h30_0_1 h32_0 h34_0_1 h36_0 h37_0
    (Xb := ![w_38_0, w_80_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h80_0])
    (D' := ![w_39_0, w_81_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h81_0])
    (D := ![w_40_0, w_82_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h82_0])
    (M := ![![w_42_0, w_84_0], ![w_44_0, w_86_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h84_0])
    (fun j ↦ by fin_cases j; exacts [h44_0, h86_0])
    (Asw := ![![w_43_0, w_85_0], ![w_45_0, w_87_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h43_0, h85_0, h45_0, h87_0])
    (Z := ![w_46_0, w_88_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h88_0])
    (E' := ![w_47_0, w_89_0]) (fun j ↦ by fin_cases j; exacts [h47_0, h89_0])
    (E := ![w_48_0, w_90_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h90_0])
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (Y := ![w_107_0, w_114_0]) (X := ![w_108_0, w_115_0])
    (K := ![![w_50_0, w_92_0], ![w_97_0, w_102_0]])
    (fun j ↦ by fin_cases j; exacts [h107_0, h114_0])
    (fun j ↦ by fin_cases j; exacts [h108_0, h115_0])
    (fun j ↦ by fin_cases j; exacts [h50_0, h92_0])
    (fun j ↦ by fin_cases j; exacts [h97_0, h102_0])
    (T := smudgeTable) (Δ' := (1 : ErrorPoly N)) hinverse
    (fun i ↦ hswitch i) (K' := ![![w_52_0, w_94_0], ![w_99_0, w_104_0]])
    (R' := ![w_111_0, w_118_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h52_0, h94_0, h99_0, h104_0])
    (fun j ↦ by fin_cases j; exacts [h111_0, h118_0])
  -- The weight `w_(o mod 20)` of the member's digit.
  simp only [concatColumns] at h54_0 h55_0 h56_0 h57_0 h58_0 h59_0 h60_0 h61_0 h62_0 h63_0 h64_0 h65_0 h66_0 h67_0 h68_0 h69_0 h70_0 h71_0 h73_0
  have hrow : ∀ c : Fin 20, w_73_0 0 c = (matrixPolynomial [smudgeWeight615 i0 c] : ExactMatrix (R i0) N 1 1) 0 0 := by
    intro c
    fin_cases c <;> simp [h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, w_53_0, w_54_0, w_55_0, w_56_0, w_57_0, w_58_0, w_59_0, w_60_0, w_61_0, w_62_0, w_63_0, w_64_0, w_65_0, w_66_0, w_67_0, w_68_0, w_69_0, w_70_0, w_71_0, w_72_0, smudgeWeight615]
  have hd : Int.fmod (Int.ofNat o) 20 = ((o % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by inst_num)]; rfl
  rw [hd] at h74_0
  have hK : w_74_0 = matrixPolynomial [smudgeWeight615 i0 ⟨o % 20, Nat.mod_lt _ (by inst_num)⟩] := by
    funext a b
    obtain rfl : a = 0 := Subsingleton.elim _ _
    obtain rfl : b = 0 := Subsingleton.elim _ _
    rw [slice_entry h74_0 (by simp; omega)]
    exact hrow _
  have hfin2 := hfin.finalizeCoefficient (by inst_num) hN L hL hLx hgq _ hK h75_0 h76_0
  have hpl := rel_monomial hfin2 (k := o / 20) (by simp only [N]; omega)
  have hq : w_78_0 = ((o / 20 : Nat) : Int) := by
    simp only [w_78_0]
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by inst_num)]; rfl
  rw [hout]
  convert hpl using 1
  all_goals simp only [w_79_0, w_96_0, w_101_0, w_106_0, w_113_0, w_120_0, w_77_0, w_95_0, w_100_0, w_105_0, w_112_0, w_119_0, hq]
  all_goals first
    | (funext a j; fin_cases a <;> fin_cases j <;> rfl)
    | (funext j; fin_cases j <;> rfl)

/-- The digit weights of the smudge LUT `669`. -/
def smudgeWeight669 (i0 : Nat) : Fin 20 → Int := ![(halfZ (i0 : Int)), 2, 8, 32, 128, 512, 2048, 8192, 32768, 131072, 524288, 2097152, 8388608, 33554432, 134217728, 536870912, 2147483648, 8589934592, 34359738368, 137438953472]

/-- The error bound added by one branch of the LUT `669`. -/
abbrev lutBranchBound669 (i0 : Nat) (B Bh : Nat) : Nat :=
  (6 + 6) * N * Bh * 16384 + (6 + 6) * N * B * 16384 + (N * 1 * ((qd i0) / 2) + (qd i0) / 2)

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- One iteration of the LUT loop `669` adds branch `i` to the accumulated relation and carries
the switching ciphertext of branch `i + 1`. -/
theorem lutStep669 {i0 : Nat} {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    [Fact (i0 < 8)] (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q0 N 1 6} {A : Fin 2 → Fin 2 → ExactMatrix Q0 N 1 6} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix (R i0) N 1 6}
    {i : Nat} (hi : i < 3) {state next : LoopState Q0 (R i0) 6}
    (hswitch : state.switch = fun k j ↦ fam k j ⟨i, by omega⟩)
    {T : Fin 2 → ExactMatrix Q0 N 1 6} {Bs : Nat} (hstate : Rel sm state.key T state.row Bs)
    {params} {o : Nat}
    (hbody : W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_671_18 Backend.backend model params i0 o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0, c 1, A 0 1, A 1 1, fam 0 0,
        fam 0 1, fam 1 0, fam 1 1, ()) next) :
    Rel sm next.key (T + branchTarget (q := Q0) (p := (qd i0)) μ sp fam ⟨i, by omega⟩) next.row
        (Bs + lutBranchBound669 i0 B Bh) ∧
      next.switch = fun k j ↦ fam k j ⟨i + 1, by omega⟩ := by
  obtain ⟨L, hL, hLx, -, hLd⟩ := layoutR i0 hi0
  unfold W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_671_18 at hbody
  extract_lets -merge _params _backend w_0_0 w_48_0 w_67_0 w_70_0 w_73_0 w_76_0 w_30_0 w_49_0 w_31_0 w_50_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_24_0 w_26_0 w_79_0 w_83_0 w_85_0 w_87_0 w_2_0 w_5_0 w_80_0 w_81_0 at hbody
  obtain ⟨w_3_0, hbody⟩ := hbody
  obtain ⟨w_6_0, hbody⟩ := hbody
  obtain ⟨w_7_0, hbody⟩ := hbody
  obtain ⟨w_8_0, hbody⟩ := hbody
  obtain ⟨w_8_concat_1, hbody⟩ := hbody
  obtain ⟨w_10_0, hbody⟩ := hbody
  obtain ⟨w_11_0, hbody⟩ := hbody
  obtain ⟨w_12_0, hbody⟩ := hbody
  obtain ⟨w_13_0, hbody⟩ := hbody
  obtain ⟨w_13_concat_1, hbody⟩ := hbody
  obtain ⟨w_14_0, hbody⟩ := hbody
  obtain ⟨w_16_0, hbody⟩ := hbody
  obtain ⟨w_18_0, hbody⟩ := hbody
  obtain ⟨w_20_0, hbody⟩ := hbody
  obtain ⟨w_21_0, hbody⟩ := hbody
  obtain ⟨w_21_concat_1, hbody⟩ := hbody
  obtain ⟨w_23_0, hbody⟩ := hbody
  obtain ⟨w_25_0, hbody⟩ := hbody
  obtain ⟨w_27_0, hbody⟩ := hbody
  obtain ⟨w_28_0, hbody⟩ := hbody
  obtain ⟨w_28_concat_1, hbody⟩ := hbody
  obtain ⟨w_29_0, hbody⟩ := hbody
  obtain ⟨w_32_0, hbody⟩ := hbody
  obtain ⟨w_33_0, hbody⟩ := hbody
  obtain ⟨w_34_0, hbody⟩ := hbody
  extract_lets -merge w_35_0 at hbody
  obtain ⟨w_36_0, hbody⟩ := hbody
  obtain ⟨w_37_0, hbody⟩ := hbody
  obtain ⟨w_38_0, hbody⟩ := hbody
  obtain ⟨w_39_0, hbody⟩ := hbody
  obtain ⟨w_40_0, hbody⟩ := hbody
  obtain ⟨w_41_0, hbody⟩ := hbody
  obtain ⟨w_42_0, hbody⟩ := hbody
  extract_lets -merge w_43_0 at hbody
  obtain ⟨w_44_0, hbody⟩ := hbody
  obtain ⟨w_45_0, hbody⟩ := hbody
  extract_lets -merge w_46_0 w_47_0 at hbody
  obtain ⟨w_51_0, hbody⟩ := hbody
  obtain ⟨w_52_0, hbody⟩ := hbody
  obtain ⟨w_53_0, hbody⟩ := hbody
  extract_lets -merge w_54_0 at hbody
  obtain ⟨w_55_0, hbody⟩ := hbody
  obtain ⟨w_56_0, hbody⟩ := hbody
  obtain ⟨w_57_0, hbody⟩ := hbody
  obtain ⟨w_58_0, hbody⟩ := hbody
  obtain ⟨w_59_0, hbody⟩ := hbody
  obtain ⟨w_60_0, hbody⟩ := hbody
  obtain ⟨w_61_0, hbody⟩ := hbody
  extract_lets -merge w_62_0 at hbody
  obtain ⟨w_63_0, hbody⟩ := hbody
  obtain ⟨w_64_0, hbody⟩ := hbody
  extract_lets -merge w_65_0 w_66_0 at hbody
  obtain ⟨w_68_0, hbody⟩ := hbody
  extract_lets -merge w_69_0 at hbody
  obtain ⟨w_71_0, hbody⟩ := hbody
  extract_lets -merge w_72_0 at hbody
  obtain ⟨w_74_0, hbody⟩ := hbody
  extract_lets -merge w_75_0 at hbody
  obtain ⟨w_77_0, hbody⟩ := hbody
  extract_lets -merge w_78_0 at hbody
  obtain ⟨w_82_0, hbody⟩ := hbody
  obtain ⟨w_84_0, hbody⟩ := hbody
  obtain ⟨w_86_0, hbody⟩ := hbody
  obtain ⟨w_88_0, hbody⟩ := hbody
  simp only [W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_671_18.constraints_0, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_671_18.constraints_1, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_671_18.constraints_2, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_671_18.constraints_3, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_671_18.constraints_4] at hbody
  obtain ⟨h2_0, h2_0_1, h3_0, -, -, h6_0, h7_0, h6_0_1, h8_0, h2_0_2, h2_0_3, h10_0, h11_0, h12_0, h11_0_1, h13_0, h14_0, -, -, h16_0, -, -, h18_0, -, -, h20_0, h18_0_1, h21_0, -, -, h23_0, -, -, h25_0, -, -, h27_0, h25_0_1, h28_0, h29_0, h32_0, h33_0, h34_0, -, -, -, -, -, -, h36_0, h37_0, -, -, -, -, -, -, h38_0, h39_0, h40_0, h41_0, h42_0, -, -, -, -, -, -, h44_0, -, -, -, -, -, -, h45_0, h51_0, h52_0, h53_0, -, -, -, -, -, -, h55_0, h56_0, -, -, -, -, -, -, h57_0, h58_0, h59_0, h60_0, h61_0, -, -, -, -, -, -, h63_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h77_0, h81_0, h81_0_1, h82_0, h81_0_2, h81_0_3, h84_0, h81_0_4, h81_0_5, h86_0, h81_0_6, h81_0_7, h88_0, hout⟩ := hbody
  have hidx : w_2_0 = Int.ofNat i := rfl
  have hnext : w_81_0 = Int.ofNat (i + 1) := by
    simp only [w_81_0, w_80_0, w_2_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h3_0 h10_0
  rw [hnext] at h82_0 h84_0 h86_0 h88_0
  obtain ⟨_, hw3⟩ := familyGet_eq h3_0
  obtain ⟨_, hw10⟩ := familyGet_eq h10_0
  obtain ⟨_, hs0⟩ := familyGet_eq h82_0
  obtain ⟨_, hs1⟩ := familyGet_eq h84_0
  obtain ⟨_, hs2⟩ := familyGet_eq h86_0
  obtain ⟨_, hs3⟩ := familyGet_eq h88_0
  subst hw3 hw10
  have hhelp' := hhelp ⟨i, by omega⟩
  have hAd : lutA model key Q0 6 2 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  simp only [hAd] at hhelp'
  have hbranch := lut_branch (n := N) (q := Q0) (q' := (R i0)) (p := (qd i0)) (m := 6) (m' := 6)
    (log := 3) (bound := 16384) (index := sigma i) hN hpow (q0_eqR i0 hi0) (by inst_num) (by inst_num)
    L hL hLx hg' hg hsm
    henc h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (bsw := state.switch) (Xb := ![w_32_0, w_51_0])
    (fun j ↦ by fin_cases j; exacts [h32_0, h51_0])
    (D' := ![w_33_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h33_0, h52_0])
    (D := ![w_34_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h34_0, h53_0])
    (M := ![![w_36_0, w_55_0], ![w_38_0, w_57_0]])
    (fun j ↦ by fin_cases j; exacts [h36_0, h55_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h57_0])
    (Asw := ![![w_37_0, w_56_0], ![w_39_0, w_58_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h37_0, h56_0, h39_0, h58_0])
    (Z := ![w_40_0, w_59_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h59_0])
    (E' := ![w_41_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h41_0, h60_0])
    (E := ![w_42_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h42_0, h61_0])
    (h := ![_, _]) (sigma_odd ⟨i, by omega⟩) hhelp' h6_0_1 h8_0 h11_0_1 h13_0 h14_0
    (Y := ![w_44_0, w_63_0]) (X := ![w_45_0, w_64_0])
    (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h44_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h45_0, h64_0])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  subst hout
  rw [hswitch] at hbranch
  refine ⟨?_, ?_⟩
  · convert hstate.add hbranch using 1
    · funext k j
      fin_cases k <;> fin_cases j <;> rfl
    · funext j
      fin_cases j <;> rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hs0, hs1, hs2, hs3]

/-- The generated loop `669` accumulates the first `count` branches and carries the switching
ciphertext of branch `count`. -/
theorem lutLoop669 {i0 : Nat} {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    [Fact (i0 < 8)] (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q0 N 1 6} {A : Fin 2 → Fin 2 → ExactMatrix Q0 N 1 6} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix (R i0) N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q0 (R i0) 6} {params} {o : Nat}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q0 (R i0) 6) ↦
      W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_671_18 Backend.backend model params i0 o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1,
          current.2.2.2.2.2.1, current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1,
          current.2.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0,
          c 1, A 0 1, A 1 1, fam 0 0, fam 0 1, fam 1 0, fam 1 1, ()) next) count
      (0, 0, 0, 0, 0, 0, fam 0 0 0, fam 0 1 0, fam 1 0 0, fam 1 1 0, ()) final) :
    Rel sm final.key (∑ i ∈ Finset.range count, lutBranchTarget (q := Q0) (p := (qd i0)) μ sp fam i)
        final.row (count * lutBranchBound669 i0 B Bh) ∧
      final.switch = fun k j ↦ fam k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q0 (R i0) 6) ↦ ∀ hk : k ≤ 3,
      Rel sm s.key (∑ i ∈ Finset.range k, lutBranchTarget (q := Q0) (p := (qd i0)) μ sp fam i) s.row
        (k * lutBranchBound669 i0 B Bh) ∧
        s.switch = fun k' j ↦ fam k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨?_, ?_⟩
    · rw [Finset.range_zero, Finset.sum_empty, zero_mul]
      have hrow : (![0, 0] : Fin 2 → ExactMatrix Q0 N 1 6) = 0 := by
        funext j
        fin_cases j <;> rfl
      simp only [LoopState.key, LoopState.row, hrow]
      exact Rel.zero
    · funext k j
      fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := lutStep669 hsm hg' hg henc hhelp (by omega) hsw hrel hbody
    refine ⟨?_, hsw'⟩
    rw [Finset.sum_range_succ, lutBranchTarget, dif_pos (by omega), add_mul, one_mul]
    exact hrel'

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem lut669 {i0 : Nat} {model : HashModel} {key : ByteArray} {params} {o : Nat} [Fact (i0 < 8)]
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix (R i0) N 1 6} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6} {outputs : Blocks (R i0) 1}
    (h : W_Stage_online_eval.parallel_parallel_generatedRoot_290_671 Backend.backend model params i0 o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11, ()) outputs)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {z : Nat}
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {B : Nat}
    (hinput : Rel sm ![![A00, A01], ![A10, A11]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * Xe (n := N) (2 * z) * unitPair sp j) • g)
      ![c0, c1] B)
    {Bh : Nat} (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    (ho : o < 160) {gq : ExactMatrix (R i0) N 1 6} (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq) {inverse : Nat}
    (hinverse : (inverse : ExactPoly (R i0) N) * ((4 : Nat) : ExactPoly (R i0) N) = 1) {BB : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient (R i0) inverse (fun k ↦ reducePoly (R i0) N (smudgeTable k)) (sigma i).toNat *
          reducePoly (R i0) N ((1 : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix (R i0) N 1 6 (e j)) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly (R i0) N (Xe (n := N) (o / 20)) • reducePoly (R i0) N
        (((smudgeWeight669 i0 ⟨o % 20, Nat.mod_lt _ (by inst_num)⟩ : Int) : ErrorPoly N) *
          ((1 : ErrorPoly N) * ((((-1 : Int) ^ (z / 4) : Int) : ErrorPoly N) *
            smudgeTable ⟨z % 4, Nat.mod_lt _ (by inst_num)⟩)) * unitPair sp j) •
        (1 : ExactMatrix (R i0) N 1 1))
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      (6 * ((3 * lutBranchBound669 i0 B Bh + ((6 + 6) * N * Bh * 16384 + (6 + 6) * N * B * 16384 +
        (N * 1 * ((qd i0) / 2) + (qd i0) / 2)) + (N + 2) * ((qd i0) / 2)) / (qd i0) + 4 * (N * 1 * BB)) * (32768 / 2)) := by
  obtain ⟨L, hL, hLx, -, hLd⟩ := layoutR i0 hi0
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_671.body at h
  extract_lets -merge _params _backend w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_73_concat_1 w_73_concat_2 w_73_concat_3 w_73_concat_4 w_73_concat_5 w_73_concat_6 w_73_concat_7 w_73_concat_8 w_73_concat_9 w_73_concat_10 w_73_concat_11 w_73_concat_12 w_73_concat_13 w_73_concat_14 w_73_concat_15 w_73_concat_16 w_73_concat_17 w_73_concat_18 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 at h
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_671.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_671.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_671.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_671.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_671.constraints_4, W_Stage_online_eval.parallel_parallel_generatedRoot_290_671.constraints_5] at h
  obtain ⟨h2_0, h4_0, h6_0, h8_0, -, h18_9, h19_0, -, -, h21_0, h22_0, h21_0_1, h23_0, h24_0, h25_0, h26_0, h25_0_1, h27_0, h28_0, -, -, h29_0, -, -, h30_0, -, -, h31_0, h30_0_1, h32_0, -, -, h33_0, -, -, h34_0, -, -, h35_0, h34_0_1, h36_0, h37_0, h38_0, h39_0, h40_0, -, -, -, -, -, -, h42_0, h43_0, -, -, -, -, -, -, h44_0, h45_0, h46_0, h47_0, h48_0, -, -, -, -, -, -, h50_0, h52_0, -, -, h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, -, -, -, -, -, -, -, -, h74_0, h75_0, h76_0, -, h80_0, h81_0, h82_0, -, -, -, -, -, -, h84_0, h85_0, -, -, -, -, -, -, h86_0, h87_0, h88_0, h89_0, h90_0, -, -, -, -, -, -, h92_0, h94_0, -, -, -, -, -, -, h97_0, h99_0, -, -, -, -, -, -, h102_0, h104_0, -, -, -, -, -, -, h107_0, -, -, -, -, -, -, h108_0, h111_0, -, -, -, -, -, -, h114_0, -, -, -, -, -, -, h115_0, h118_0, hout⟩ := h
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  rw [hwa, hwb, hwc, hwd] at h18_9
  rw [show Int.toNat 3 = 3 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := lutLoop669 (fam := fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j)
    (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]])
    (final := (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) hsm hg' hg hinput hhelp (count := 3) (by decide) h18_9
  obtain ⟨_, hw19⟩ := familyGet_eq (k := 3) h19_0
  obtain ⟨_, hw24⟩ := familyGet_eq (k := 3) h24_0
  rw [hw19] at h21_0_1
  rw [hw24] at h25_0_1
  have hA0 : w_21_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h21_0
  have hA1 : w_25_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h25_0
  have hA2 : w_22_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h22_0
  have hA3 : w_26_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h26_0
  rw [hA0] at h21_0_1
  rw [hA2] at h23_0
  rw [hA1] at h25_0_1
  rw [hA3] at h27_0
  have hhelp3 := hhelp 3
  have hfin := lut_round (q := Q0) (q' := (R i0)) (p := (qd i0)) (m := 6) (m' := 6) (q0_eqR i0 hi0) (by inst_num)
    (by inst_num) (by inst_num) L hL hLx hsm hg' hg hinput hloop hloopSwitch hhelp3
    h29_0 h30_0 h31_0 h33_0 h34_0 h35_0 h30_0_1 h32_0 h34_0_1 h36_0 h37_0
    (Xb := ![w_38_0, w_80_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h80_0])
    (D' := ![w_39_0, w_81_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h81_0])
    (D := ![w_40_0, w_82_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h82_0])
    (M := ![![w_42_0, w_84_0], ![w_44_0, w_86_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h84_0])
    (fun j ↦ by fin_cases j; exacts [h44_0, h86_0])
    (Asw := ![![w_43_0, w_85_0], ![w_45_0, w_87_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h43_0, h85_0, h45_0, h87_0])
    (Z := ![w_46_0, w_88_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h88_0])
    (E' := ![w_47_0, w_89_0]) (fun j ↦ by fin_cases j; exacts [h47_0, h89_0])
    (E := ![w_48_0, w_90_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h90_0])
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (Y := ![w_107_0, w_114_0]) (X := ![w_108_0, w_115_0])
    (K := ![![w_50_0, w_92_0], ![w_97_0, w_102_0]])
    (fun j ↦ by fin_cases j; exacts [h107_0, h114_0])
    (fun j ↦ by fin_cases j; exacts [h108_0, h115_0])
    (fun j ↦ by fin_cases j; exacts [h50_0, h92_0])
    (fun j ↦ by fin_cases j; exacts [h97_0, h102_0])
    (T := smudgeTable) (Δ' := (1 : ErrorPoly N)) hinverse
    (fun i ↦ hswitch i) (K' := ![![w_52_0, w_94_0], ![w_99_0, w_104_0]])
    (R' := ![w_111_0, w_118_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h52_0, h94_0, h99_0, h104_0])
    (fun j ↦ by fin_cases j; exacts [h111_0, h118_0])
  -- The weight `w_(o mod 20)` of the member's digit.
  simp only [concatColumns] at h54_0 h55_0 h56_0 h57_0 h58_0 h59_0 h60_0 h61_0 h62_0 h63_0 h64_0 h65_0 h66_0 h67_0 h68_0 h69_0 h70_0 h71_0 h73_0
  have hrow : ∀ c : Fin 20, w_73_0 0 c = (matrixPolynomial [smudgeWeight669 i0 c] : ExactMatrix (R i0) N 1 1) 0 0 := by
    intro c
    fin_cases c <;> simp [h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, w_53_0, w_54_0, w_55_0, w_56_0, w_57_0, w_58_0, w_59_0, w_60_0, w_61_0, w_62_0, w_63_0, w_64_0, w_65_0, w_66_0, w_67_0, w_68_0, w_69_0, w_70_0, w_71_0, w_72_0, smudgeWeight669]
  have hd : Int.fmod (Int.ofNat o) 20 = ((o % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by inst_num)]; rfl
  rw [hd] at h74_0
  have hK : w_74_0 = matrixPolynomial [smudgeWeight669 i0 ⟨o % 20, Nat.mod_lt _ (by inst_num)⟩] := by
    funext a b
    obtain rfl : a = 0 := Subsingleton.elim _ _
    obtain rfl : b = 0 := Subsingleton.elim _ _
    rw [slice_entry h74_0 (by simp; omega)]
    exact hrow _
  have hfin2 := hfin.finalizeCoefficient (by inst_num) hN L hL hLx hgq _ hK h75_0 h76_0
  have hpl := rel_monomial hfin2 (k := o / 20) (by simp only [N]; omega)
  have hq : w_78_0 = ((o / 20 : Nat) : Int) := by
    simp only [w_78_0]
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by inst_num)]; rfl
  rw [hout]
  convert hpl using 1
  all_goals simp only [w_79_0, w_96_0, w_101_0, w_106_0, w_113_0, w_120_0, w_77_0, w_95_0, w_100_0, w_105_0, w_112_0, w_119_0, hq]
  all_goals first
    | (funext a j; fin_cases a <;> fin_cases j <;> rfl)
    | (funext j; fin_cases j <;> rfl)

/-- The digit weights of the smudge LUT `723`. -/
def smudgeWeight723 (i0 : Nat) : Fin 20 → Int := ![(halfZ (i0 : Int)), 2, 8, 32, 128, 512, 2048, 8192, 32768, 131072, 524288, 2097152, 8388608, 33554432, 134217728, 536870912, 2147483648, 8589934592, 34359738368, 137438953472]

/-- The error bound added by one branch of the LUT `723`. -/
abbrev lutBranchBound723 (i0 : Nat) (B Bh : Nat) : Nat :=
  (6 + 6) * N * Bh * 16384 + (6 + 6) * N * B * 16384 + (N * 1 * ((qd i0) / 2) + (qd i0) / 2)

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- One iteration of the LUT loop `723` adds branch `i` to the accumulated relation and carries
the switching ciphertext of branch `i + 1`. -/
theorem lutStep723 {i0 : Nat} {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    [Fact (i0 < 8)] (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q0 N 1 6} {A : Fin 2 → Fin 2 → ExactMatrix Q0 N 1 6} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix (R i0) N 1 6}
    {i : Nat} (hi : i < 3) {state next : LoopState Q0 (R i0) 6}
    (hswitch : state.switch = fun k j ↦ fam k j ⟨i, by omega⟩)
    {T : Fin 2 → ExactMatrix Q0 N 1 6} {Bs : Nat} (hstate : Rel sm state.key T state.row Bs)
    {params} {o : Nat}
    (hbody : W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_725_18 Backend.backend model params i0 o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0, c 1, A 0 1, A 1 1, fam 0 0,
        fam 0 1, fam 1 0, fam 1 1, ()) next) :
    Rel sm next.key (T + branchTarget (q := Q0) (p := (qd i0)) μ sp fam ⟨i, by omega⟩) next.row
        (Bs + lutBranchBound723 i0 B Bh) ∧
      next.switch = fun k j ↦ fam k j ⟨i + 1, by omega⟩ := by
  obtain ⟨L, hL, hLx, -, hLd⟩ := layoutR i0 hi0
  unfold W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_725_18 at hbody
  extract_lets -merge _params _backend w_0_0 w_48_0 w_67_0 w_70_0 w_73_0 w_76_0 w_30_0 w_49_0 w_31_0 w_50_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_24_0 w_26_0 w_79_0 w_83_0 w_85_0 w_87_0 w_2_0 w_5_0 w_80_0 w_81_0 at hbody
  obtain ⟨w_3_0, hbody⟩ := hbody
  obtain ⟨w_6_0, hbody⟩ := hbody
  obtain ⟨w_7_0, hbody⟩ := hbody
  obtain ⟨w_8_0, hbody⟩ := hbody
  obtain ⟨w_8_concat_1, hbody⟩ := hbody
  obtain ⟨w_10_0, hbody⟩ := hbody
  obtain ⟨w_11_0, hbody⟩ := hbody
  obtain ⟨w_12_0, hbody⟩ := hbody
  obtain ⟨w_13_0, hbody⟩ := hbody
  obtain ⟨w_13_concat_1, hbody⟩ := hbody
  obtain ⟨w_14_0, hbody⟩ := hbody
  obtain ⟨w_16_0, hbody⟩ := hbody
  obtain ⟨w_18_0, hbody⟩ := hbody
  obtain ⟨w_20_0, hbody⟩ := hbody
  obtain ⟨w_21_0, hbody⟩ := hbody
  obtain ⟨w_21_concat_1, hbody⟩ := hbody
  obtain ⟨w_23_0, hbody⟩ := hbody
  obtain ⟨w_25_0, hbody⟩ := hbody
  obtain ⟨w_27_0, hbody⟩ := hbody
  obtain ⟨w_28_0, hbody⟩ := hbody
  obtain ⟨w_28_concat_1, hbody⟩ := hbody
  obtain ⟨w_29_0, hbody⟩ := hbody
  obtain ⟨w_32_0, hbody⟩ := hbody
  obtain ⟨w_33_0, hbody⟩ := hbody
  obtain ⟨w_34_0, hbody⟩ := hbody
  extract_lets -merge w_35_0 at hbody
  obtain ⟨w_36_0, hbody⟩ := hbody
  obtain ⟨w_37_0, hbody⟩ := hbody
  obtain ⟨w_38_0, hbody⟩ := hbody
  obtain ⟨w_39_0, hbody⟩ := hbody
  obtain ⟨w_40_0, hbody⟩ := hbody
  obtain ⟨w_41_0, hbody⟩ := hbody
  obtain ⟨w_42_0, hbody⟩ := hbody
  extract_lets -merge w_43_0 at hbody
  obtain ⟨w_44_0, hbody⟩ := hbody
  obtain ⟨w_45_0, hbody⟩ := hbody
  extract_lets -merge w_46_0 w_47_0 at hbody
  obtain ⟨w_51_0, hbody⟩ := hbody
  obtain ⟨w_52_0, hbody⟩ := hbody
  obtain ⟨w_53_0, hbody⟩ := hbody
  extract_lets -merge w_54_0 at hbody
  obtain ⟨w_55_0, hbody⟩ := hbody
  obtain ⟨w_56_0, hbody⟩ := hbody
  obtain ⟨w_57_0, hbody⟩ := hbody
  obtain ⟨w_58_0, hbody⟩ := hbody
  obtain ⟨w_59_0, hbody⟩ := hbody
  obtain ⟨w_60_0, hbody⟩ := hbody
  obtain ⟨w_61_0, hbody⟩ := hbody
  extract_lets -merge w_62_0 at hbody
  obtain ⟨w_63_0, hbody⟩ := hbody
  obtain ⟨w_64_0, hbody⟩ := hbody
  extract_lets -merge w_65_0 w_66_0 at hbody
  obtain ⟨w_68_0, hbody⟩ := hbody
  extract_lets -merge w_69_0 at hbody
  obtain ⟨w_71_0, hbody⟩ := hbody
  extract_lets -merge w_72_0 at hbody
  obtain ⟨w_74_0, hbody⟩ := hbody
  extract_lets -merge w_75_0 at hbody
  obtain ⟨w_77_0, hbody⟩ := hbody
  extract_lets -merge w_78_0 at hbody
  obtain ⟨w_82_0, hbody⟩ := hbody
  obtain ⟨w_84_0, hbody⟩ := hbody
  obtain ⟨w_86_0, hbody⟩ := hbody
  obtain ⟨w_88_0, hbody⟩ := hbody
  simp only [W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_725_18.constraints_0, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_725_18.constraints_1, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_725_18.constraints_2, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_725_18.constraints_3, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_725_18.constraints_4] at hbody
  obtain ⟨h2_0, h2_0_1, h3_0, -, -, h6_0, h7_0, h6_0_1, h8_0, h2_0_2, h2_0_3, h10_0, h11_0, h12_0, h11_0_1, h13_0, h14_0, -, -, h16_0, -, -, h18_0, -, -, h20_0, h18_0_1, h21_0, -, -, h23_0, -, -, h25_0, -, -, h27_0, h25_0_1, h28_0, h29_0, h32_0, h33_0, h34_0, -, -, -, -, -, -, h36_0, h37_0, -, -, -, -, -, -, h38_0, h39_0, h40_0, h41_0, h42_0, -, -, -, -, -, -, h44_0, -, -, -, -, -, -, h45_0, h51_0, h52_0, h53_0, -, -, -, -, -, -, h55_0, h56_0, -, -, -, -, -, -, h57_0, h58_0, h59_0, h60_0, h61_0, -, -, -, -, -, -, h63_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h77_0, h81_0, h81_0_1, h82_0, h81_0_2, h81_0_3, h84_0, h81_0_4, h81_0_5, h86_0, h81_0_6, h81_0_7, h88_0, hout⟩ := hbody
  have hidx : w_2_0 = Int.ofNat i := rfl
  have hnext : w_81_0 = Int.ofNat (i + 1) := by
    simp only [w_81_0, w_80_0, w_2_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h3_0 h10_0
  rw [hnext] at h82_0 h84_0 h86_0 h88_0
  obtain ⟨_, hw3⟩ := familyGet_eq h3_0
  obtain ⟨_, hw10⟩ := familyGet_eq h10_0
  obtain ⟨_, hs0⟩ := familyGet_eq h82_0
  obtain ⟨_, hs1⟩ := familyGet_eq h84_0
  obtain ⟨_, hs2⟩ := familyGet_eq h86_0
  obtain ⟨_, hs3⟩ := familyGet_eq h88_0
  subst hw3 hw10
  have hhelp' := hhelp ⟨i, by omega⟩
  have hAd : lutA model key Q0 6 2 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  simp only [hAd] at hhelp'
  have hbranch := lut_branch (n := N) (q := Q0) (q' := (R i0)) (p := (qd i0)) (m := 6) (m' := 6)
    (log := 3) (bound := 16384) (index := sigma i) hN hpow (q0_eqR i0 hi0) (by inst_num) (by inst_num)
    L hL hLx hg' hg hsm
    henc h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (bsw := state.switch) (Xb := ![w_32_0, w_51_0])
    (fun j ↦ by fin_cases j; exacts [h32_0, h51_0])
    (D' := ![w_33_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h33_0, h52_0])
    (D := ![w_34_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h34_0, h53_0])
    (M := ![![w_36_0, w_55_0], ![w_38_0, w_57_0]])
    (fun j ↦ by fin_cases j; exacts [h36_0, h55_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h57_0])
    (Asw := ![![w_37_0, w_56_0], ![w_39_0, w_58_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h37_0, h56_0, h39_0, h58_0])
    (Z := ![w_40_0, w_59_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h59_0])
    (E' := ![w_41_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h41_0, h60_0])
    (E := ![w_42_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h42_0, h61_0])
    (h := ![_, _]) (sigma_odd ⟨i, by omega⟩) hhelp' h6_0_1 h8_0 h11_0_1 h13_0 h14_0
    (Y := ![w_44_0, w_63_0]) (X := ![w_45_0, w_64_0])
    (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h44_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h45_0, h64_0])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  subst hout
  rw [hswitch] at hbranch
  refine ⟨?_, ?_⟩
  · convert hstate.add hbranch using 1
    · funext k j
      fin_cases k <;> fin_cases j <;> rfl
    · funext j
      fin_cases j <;> rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hs0, hs1, hs2, hs3]

/-- The generated loop `723` accumulates the first `count` branches and carries the switching
ciphertext of branch `count`. -/
theorem lutLoop723 {i0 : Nat} {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    [Fact (i0 < 8)] (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q0 N 1 6} {A : Fin 2 → Fin 2 → ExactMatrix Q0 N 1 6} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix (R i0) N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q0 (R i0) 6} {params} {o : Nat}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q0 (R i0) 6) ↦
      W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_725_18 Backend.backend model params i0 o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1,
          current.2.2.2.2.2.1, current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1,
          current.2.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0,
          c 1, A 0 1, A 1 1, fam 0 0, fam 0 1, fam 1 0, fam 1 1, ()) next) count
      (0, 0, 0, 0, 0, 0, fam 0 0 0, fam 0 1 0, fam 1 0 0, fam 1 1 0, ()) final) :
    Rel sm final.key (∑ i ∈ Finset.range count, lutBranchTarget (q := Q0) (p := (qd i0)) μ sp fam i)
        final.row (count * lutBranchBound723 i0 B Bh) ∧
      final.switch = fun k j ↦ fam k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q0 (R i0) 6) ↦ ∀ hk : k ≤ 3,
      Rel sm s.key (∑ i ∈ Finset.range k, lutBranchTarget (q := Q0) (p := (qd i0)) μ sp fam i) s.row
        (k * lutBranchBound723 i0 B Bh) ∧
        s.switch = fun k' j ↦ fam k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨?_, ?_⟩
    · rw [Finset.range_zero, Finset.sum_empty, zero_mul]
      have hrow : (![0, 0] : Fin 2 → ExactMatrix Q0 N 1 6) = 0 := by
        funext j
        fin_cases j <;> rfl
      simp only [LoopState.key, LoopState.row, hrow]
      exact Rel.zero
    · funext k j
      fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := lutStep723 hsm hg' hg henc hhelp (by omega) hsw hrel hbody
    refine ⟨?_, hsw'⟩
    rw [Finset.sum_range_succ, lutBranchTarget, dif_pos (by omega), add_mul, one_mul]
    exact hrel'

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem lut723 {i0 : Nat} {model : HashModel} {key : ByteArray} {params} {o : Nat} [Fact (i0 < 8)]
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix (R i0) N 1 6} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6} {outputs : Blocks (R i0) 1}
    (h : W_Stage_online_eval.parallel_parallel_generatedRoot_290_725 Backend.backend model params i0 o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11, ()) outputs)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {z : Nat}
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {B : Nat}
    (hinput : Rel sm ![![A00, A01], ![A10, A11]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * Xe (n := N) (2 * z) * unitPair sp j) • g)
      ![c0, c1] B)
    {Bh : Nat} (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    (ho : o < 160) {gq : ExactMatrix (R i0) N 1 6} (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq) {inverse : Nat}
    (hinverse : (inverse : ExactPoly (R i0) N) * ((4 : Nat) : ExactPoly (R i0) N) = 1) {BB : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient (R i0) inverse (fun k ↦ reducePoly (R i0) N (smudgeTable k)) (sigma i).toNat *
          reducePoly (R i0) N ((1 : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix (R i0) N 1 6 (e j)) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly (R i0) N (Xe (n := N) (o / 20)) • reducePoly (R i0) N
        (((smudgeWeight723 i0 ⟨o % 20, Nat.mod_lt _ (by inst_num)⟩ : Int) : ErrorPoly N) *
          ((1 : ErrorPoly N) * ((((-1 : Int) ^ (z / 4) : Int) : ErrorPoly N) *
            smudgeTable ⟨z % 4, Nat.mod_lt _ (by inst_num)⟩)) * unitPair sp j) •
        (1 : ExactMatrix (R i0) N 1 1))
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      (6 * ((3 * lutBranchBound723 i0 B Bh + ((6 + 6) * N * Bh * 16384 + (6 + 6) * N * B * 16384 +
        (N * 1 * ((qd i0) / 2) + (qd i0) / 2)) + (N + 2) * ((qd i0) / 2)) / (qd i0) + 4 * (N * 1 * BB)) * (32768 / 2)) := by
  obtain ⟨L, hL, hLx, -, hLd⟩ := layoutR i0 hi0
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_725.body at h
  extract_lets -merge _params _backend w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_73_concat_1 w_73_concat_2 w_73_concat_3 w_73_concat_4 w_73_concat_5 w_73_concat_6 w_73_concat_7 w_73_concat_8 w_73_concat_9 w_73_concat_10 w_73_concat_11 w_73_concat_12 w_73_concat_13 w_73_concat_14 w_73_concat_15 w_73_concat_16 w_73_concat_17 w_73_concat_18 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 at h
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_725.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_725.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_725.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_725.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_725.constraints_4, W_Stage_online_eval.parallel_parallel_generatedRoot_290_725.constraints_5] at h
  obtain ⟨h2_0, h4_0, h6_0, h8_0, -, h18_9, h19_0, -, -, h21_0, h22_0, h21_0_1, h23_0, h24_0, h25_0, h26_0, h25_0_1, h27_0, h28_0, -, -, h29_0, -, -, h30_0, -, -, h31_0, h30_0_1, h32_0, -, -, h33_0, -, -, h34_0, -, -, h35_0, h34_0_1, h36_0, h37_0, h38_0, h39_0, h40_0, -, -, -, -, -, -, h42_0, h43_0, -, -, -, -, -, -, h44_0, h45_0, h46_0, h47_0, h48_0, -, -, -, -, -, -, h50_0, h52_0, -, -, h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, -, -, -, -, -, -, -, -, h74_0, h75_0, h76_0, -, h80_0, h81_0, h82_0, -, -, -, -, -, -, h84_0, h85_0, -, -, -, -, -, -, h86_0, h87_0, h88_0, h89_0, h90_0, -, -, -, -, -, -, h92_0, h94_0, -, -, -, -, -, -, h97_0, h99_0, -, -, -, -, -, -, h102_0, h104_0, -, -, -, -, -, -, h107_0, -, -, -, -, -, -, h108_0, h111_0, -, -, -, -, -, -, h114_0, -, -, -, -, -, -, h115_0, h118_0, hout⟩ := h
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  rw [hwa, hwb, hwc, hwd] at h18_9
  rw [show Int.toNat 3 = 3 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := lutLoop723 (fam := fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j)
    (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]])
    (final := (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) hsm hg' hg hinput hhelp (count := 3) (by decide) h18_9
  obtain ⟨_, hw19⟩ := familyGet_eq (k := 3) h19_0
  obtain ⟨_, hw24⟩ := familyGet_eq (k := 3) h24_0
  rw [hw19] at h21_0_1
  rw [hw24] at h25_0_1
  have hA0 : w_21_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h21_0
  have hA1 : w_25_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h25_0
  have hA2 : w_22_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h22_0
  have hA3 : w_26_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h26_0
  rw [hA0] at h21_0_1
  rw [hA2] at h23_0
  rw [hA1] at h25_0_1
  rw [hA3] at h27_0
  have hhelp3 := hhelp 3
  have hfin := lut_round (q := Q0) (q' := (R i0)) (p := (qd i0)) (m := 6) (m' := 6) (q0_eqR i0 hi0) (by inst_num)
    (by inst_num) (by inst_num) L hL hLx hsm hg' hg hinput hloop hloopSwitch hhelp3
    h29_0 h30_0 h31_0 h33_0 h34_0 h35_0 h30_0_1 h32_0 h34_0_1 h36_0 h37_0
    (Xb := ![w_38_0, w_80_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h80_0])
    (D' := ![w_39_0, w_81_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h81_0])
    (D := ![w_40_0, w_82_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h82_0])
    (M := ![![w_42_0, w_84_0], ![w_44_0, w_86_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h84_0])
    (fun j ↦ by fin_cases j; exacts [h44_0, h86_0])
    (Asw := ![![w_43_0, w_85_0], ![w_45_0, w_87_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h43_0, h85_0, h45_0, h87_0])
    (Z := ![w_46_0, w_88_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h88_0])
    (E' := ![w_47_0, w_89_0]) (fun j ↦ by fin_cases j; exacts [h47_0, h89_0])
    (E := ![w_48_0, w_90_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h90_0])
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (Y := ![w_107_0, w_114_0]) (X := ![w_108_0, w_115_0])
    (K := ![![w_50_0, w_92_0], ![w_97_0, w_102_0]])
    (fun j ↦ by fin_cases j; exacts [h107_0, h114_0])
    (fun j ↦ by fin_cases j; exacts [h108_0, h115_0])
    (fun j ↦ by fin_cases j; exacts [h50_0, h92_0])
    (fun j ↦ by fin_cases j; exacts [h97_0, h102_0])
    (T := smudgeTable) (Δ' := (1 : ErrorPoly N)) hinverse
    (fun i ↦ hswitch i) (K' := ![![w_52_0, w_94_0], ![w_99_0, w_104_0]])
    (R' := ![w_111_0, w_118_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h52_0, h94_0, h99_0, h104_0])
    (fun j ↦ by fin_cases j; exacts [h111_0, h118_0])
  -- The weight `w_(o mod 20)` of the member's digit.
  simp only [concatColumns] at h54_0 h55_0 h56_0 h57_0 h58_0 h59_0 h60_0 h61_0 h62_0 h63_0 h64_0 h65_0 h66_0 h67_0 h68_0 h69_0 h70_0 h71_0 h73_0
  have hrow : ∀ c : Fin 20, w_73_0 0 c = (matrixPolynomial [smudgeWeight723 i0 c] : ExactMatrix (R i0) N 1 1) 0 0 := by
    intro c
    fin_cases c <;> simp [h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, w_53_0, w_54_0, w_55_0, w_56_0, w_57_0, w_58_0, w_59_0, w_60_0, w_61_0, w_62_0, w_63_0, w_64_0, w_65_0, w_66_0, w_67_0, w_68_0, w_69_0, w_70_0, w_71_0, w_72_0, smudgeWeight723]
  have hd : Int.fmod (Int.ofNat o) 20 = ((o % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by inst_num)]; rfl
  rw [hd] at h74_0
  have hK : w_74_0 = matrixPolynomial [smudgeWeight723 i0 ⟨o % 20, Nat.mod_lt _ (by inst_num)⟩] := by
    funext a b
    obtain rfl : a = 0 := Subsingleton.elim _ _
    obtain rfl : b = 0 := Subsingleton.elim _ _
    rw [slice_entry h74_0 (by simp; omega)]
    exact hrow _
  have hfin2 := hfin.finalizeCoefficient (by inst_num) hN L hL hLx hgq _ hK h75_0 h76_0
  have hpl := rel_monomial hfin2 (k := o / 20) (by simp only [N]; omega)
  have hq : w_78_0 = ((o / 20 : Nat) : Int) := by
    simp only [w_78_0]
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by inst_num)]; rfl
  rw [hout]
  convert hpl using 1
  all_goals simp only [w_79_0, w_96_0, w_101_0, w_106_0, w_113_0, w_120_0, w_77_0, w_95_0, w_100_0, w_105_0, w_112_0, w_119_0, hq]
  all_goals first
    | (funext a j; fin_cases a <;> fin_cases j <;> rfl)
    | (funext j; fin_cases j <;> rfl)

/-- The digit weights of the smudge LUT `777`. -/
def smudgeWeight777 (i0 : Nat) : Fin 20 → Int := ![(halfZ (i0 : Int)), 2, 8, 32, 128, 512, 2048, 8192, 32768, 131072, 524288, 2097152, 8388608, 33554432, 134217728, 536870912, 2147483648, 8589934592, 34359738368, 137438953472]

/-- The error bound added by one branch of the LUT `777`. -/
abbrev lutBranchBound777 (i0 : Nat) (B Bh : Nat) : Nat :=
  (6 + 6) * N * Bh * 16384 + (6 + 6) * N * B * 16384 + (N * 1 * ((qd i0) / 2) + (qd i0) / 2)

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- One iteration of the LUT loop `777` adds branch `i` to the accumulated relation and carries
the switching ciphertext of branch `i + 1`. -/
theorem lutStep777 {i0 : Nat} {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    [Fact (i0 < 8)] (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q0 N 1 6} {A : Fin 2 → Fin 2 → ExactMatrix Q0 N 1 6} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix (R i0) N 1 6}
    {i : Nat} (hi : i < 3) {state next : LoopState Q0 (R i0) 6}
    (hswitch : state.switch = fun k j ↦ fam k j ⟨i, by omega⟩)
    {T : Fin 2 → ExactMatrix Q0 N 1 6} {Bs : Nat} (hstate : Rel sm state.key T state.row Bs)
    {params} {o : Nat}
    (hbody : W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_779_18 Backend.backend model params i0 o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0, c 1, A 0 1, A 1 1, fam 0 0,
        fam 0 1, fam 1 0, fam 1 1, ()) next) :
    Rel sm next.key (T + branchTarget (q := Q0) (p := (qd i0)) μ sp fam ⟨i, by omega⟩) next.row
        (Bs + lutBranchBound777 i0 B Bh) ∧
      next.switch = fun k j ↦ fam k j ⟨i + 1, by omega⟩ := by
  obtain ⟨L, hL, hLx, -, hLd⟩ := layoutR i0 hi0
  unfold W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_779_18 at hbody
  extract_lets -merge _params _backend w_0_0 w_48_0 w_67_0 w_70_0 w_73_0 w_76_0 w_30_0 w_49_0 w_31_0 w_50_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_24_0 w_26_0 w_79_0 w_83_0 w_85_0 w_87_0 w_2_0 w_5_0 w_80_0 w_81_0 at hbody
  obtain ⟨w_3_0, hbody⟩ := hbody
  obtain ⟨w_6_0, hbody⟩ := hbody
  obtain ⟨w_7_0, hbody⟩ := hbody
  obtain ⟨w_8_0, hbody⟩ := hbody
  obtain ⟨w_8_concat_1, hbody⟩ := hbody
  obtain ⟨w_10_0, hbody⟩ := hbody
  obtain ⟨w_11_0, hbody⟩ := hbody
  obtain ⟨w_12_0, hbody⟩ := hbody
  obtain ⟨w_13_0, hbody⟩ := hbody
  obtain ⟨w_13_concat_1, hbody⟩ := hbody
  obtain ⟨w_14_0, hbody⟩ := hbody
  obtain ⟨w_16_0, hbody⟩ := hbody
  obtain ⟨w_18_0, hbody⟩ := hbody
  obtain ⟨w_20_0, hbody⟩ := hbody
  obtain ⟨w_21_0, hbody⟩ := hbody
  obtain ⟨w_21_concat_1, hbody⟩ := hbody
  obtain ⟨w_23_0, hbody⟩ := hbody
  obtain ⟨w_25_0, hbody⟩ := hbody
  obtain ⟨w_27_0, hbody⟩ := hbody
  obtain ⟨w_28_0, hbody⟩ := hbody
  obtain ⟨w_28_concat_1, hbody⟩ := hbody
  obtain ⟨w_29_0, hbody⟩ := hbody
  obtain ⟨w_32_0, hbody⟩ := hbody
  obtain ⟨w_33_0, hbody⟩ := hbody
  obtain ⟨w_34_0, hbody⟩ := hbody
  extract_lets -merge w_35_0 at hbody
  obtain ⟨w_36_0, hbody⟩ := hbody
  obtain ⟨w_37_0, hbody⟩ := hbody
  obtain ⟨w_38_0, hbody⟩ := hbody
  obtain ⟨w_39_0, hbody⟩ := hbody
  obtain ⟨w_40_0, hbody⟩ := hbody
  obtain ⟨w_41_0, hbody⟩ := hbody
  obtain ⟨w_42_0, hbody⟩ := hbody
  extract_lets -merge w_43_0 at hbody
  obtain ⟨w_44_0, hbody⟩ := hbody
  obtain ⟨w_45_0, hbody⟩ := hbody
  extract_lets -merge w_46_0 w_47_0 at hbody
  obtain ⟨w_51_0, hbody⟩ := hbody
  obtain ⟨w_52_0, hbody⟩ := hbody
  obtain ⟨w_53_0, hbody⟩ := hbody
  extract_lets -merge w_54_0 at hbody
  obtain ⟨w_55_0, hbody⟩ := hbody
  obtain ⟨w_56_0, hbody⟩ := hbody
  obtain ⟨w_57_0, hbody⟩ := hbody
  obtain ⟨w_58_0, hbody⟩ := hbody
  obtain ⟨w_59_0, hbody⟩ := hbody
  obtain ⟨w_60_0, hbody⟩ := hbody
  obtain ⟨w_61_0, hbody⟩ := hbody
  extract_lets -merge w_62_0 at hbody
  obtain ⟨w_63_0, hbody⟩ := hbody
  obtain ⟨w_64_0, hbody⟩ := hbody
  extract_lets -merge w_65_0 w_66_0 at hbody
  obtain ⟨w_68_0, hbody⟩ := hbody
  extract_lets -merge w_69_0 at hbody
  obtain ⟨w_71_0, hbody⟩ := hbody
  extract_lets -merge w_72_0 at hbody
  obtain ⟨w_74_0, hbody⟩ := hbody
  extract_lets -merge w_75_0 at hbody
  obtain ⟨w_77_0, hbody⟩ := hbody
  extract_lets -merge w_78_0 at hbody
  obtain ⟨w_82_0, hbody⟩ := hbody
  obtain ⟨w_84_0, hbody⟩ := hbody
  obtain ⟨w_86_0, hbody⟩ := hbody
  obtain ⟨w_88_0, hbody⟩ := hbody
  simp only [W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_779_18.constraints_0, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_779_18.constraints_1, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_779_18.constraints_2, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_779_18.constraints_3, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_779_18.constraints_4] at hbody
  obtain ⟨h2_0, h2_0_1, h3_0, -, -, h6_0, h7_0, h6_0_1, h8_0, h2_0_2, h2_0_3, h10_0, h11_0, h12_0, h11_0_1, h13_0, h14_0, -, -, h16_0, -, -, h18_0, -, -, h20_0, h18_0_1, h21_0, -, -, h23_0, -, -, h25_0, -, -, h27_0, h25_0_1, h28_0, h29_0, h32_0, h33_0, h34_0, -, -, -, -, -, -, h36_0, h37_0, -, -, -, -, -, -, h38_0, h39_0, h40_0, h41_0, h42_0, -, -, -, -, -, -, h44_0, -, -, -, -, -, -, h45_0, h51_0, h52_0, h53_0, -, -, -, -, -, -, h55_0, h56_0, -, -, -, -, -, -, h57_0, h58_0, h59_0, h60_0, h61_0, -, -, -, -, -, -, h63_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h77_0, h81_0, h81_0_1, h82_0, h81_0_2, h81_0_3, h84_0, h81_0_4, h81_0_5, h86_0, h81_0_6, h81_0_7, h88_0, hout⟩ := hbody
  have hidx : w_2_0 = Int.ofNat i := rfl
  have hnext : w_81_0 = Int.ofNat (i + 1) := by
    simp only [w_81_0, w_80_0, w_2_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h3_0 h10_0
  rw [hnext] at h82_0 h84_0 h86_0 h88_0
  obtain ⟨_, hw3⟩ := familyGet_eq h3_0
  obtain ⟨_, hw10⟩ := familyGet_eq h10_0
  obtain ⟨_, hs0⟩ := familyGet_eq h82_0
  obtain ⟨_, hs1⟩ := familyGet_eq h84_0
  obtain ⟨_, hs2⟩ := familyGet_eq h86_0
  obtain ⟨_, hs3⟩ := familyGet_eq h88_0
  subst hw3 hw10
  have hhelp' := hhelp ⟨i, by omega⟩
  have hAd : lutA model key Q0 6 2 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  simp only [hAd] at hhelp'
  have hbranch := lut_branch (n := N) (q := Q0) (q' := (R i0)) (p := (qd i0)) (m := 6) (m' := 6)
    (log := 3) (bound := 16384) (index := sigma i) hN hpow (q0_eqR i0 hi0) (by inst_num) (by inst_num)
    L hL hLx hg' hg hsm
    henc h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (bsw := state.switch) (Xb := ![w_32_0, w_51_0])
    (fun j ↦ by fin_cases j; exacts [h32_0, h51_0])
    (D' := ![w_33_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h33_0, h52_0])
    (D := ![w_34_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h34_0, h53_0])
    (M := ![![w_36_0, w_55_0], ![w_38_0, w_57_0]])
    (fun j ↦ by fin_cases j; exacts [h36_0, h55_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h57_0])
    (Asw := ![![w_37_0, w_56_0], ![w_39_0, w_58_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h37_0, h56_0, h39_0, h58_0])
    (Z := ![w_40_0, w_59_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h59_0])
    (E' := ![w_41_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h41_0, h60_0])
    (E := ![w_42_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h42_0, h61_0])
    (h := ![_, _]) (sigma_odd ⟨i, by omega⟩) hhelp' h6_0_1 h8_0 h11_0_1 h13_0 h14_0
    (Y := ![w_44_0, w_63_0]) (X := ![w_45_0, w_64_0])
    (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h44_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h45_0, h64_0])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  subst hout
  rw [hswitch] at hbranch
  refine ⟨?_, ?_⟩
  · convert hstate.add hbranch using 1
    · funext k j
      fin_cases k <;> fin_cases j <;> rfl
    · funext j
      fin_cases j <;> rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hs0, hs1, hs2, hs3]

/-- The generated loop `777` accumulates the first `count` branches and carries the switching
ciphertext of branch `count`. -/
theorem lutLoop777 {i0 : Nat} {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    [Fact (i0 < 8)] (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q0 N 1 6} {A : Fin 2 → Fin 2 → ExactMatrix Q0 N 1 6} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix (R i0) N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q0 (R i0) 6} {params} {o : Nat}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q0 (R i0) 6) ↦
      W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_779_18 Backend.backend model params i0 o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1,
          current.2.2.2.2.2.1, current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1,
          current.2.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0,
          c 1, A 0 1, A 1 1, fam 0 0, fam 0 1, fam 1 0, fam 1 1, ()) next) count
      (0, 0, 0, 0, 0, 0, fam 0 0 0, fam 0 1 0, fam 1 0 0, fam 1 1 0, ()) final) :
    Rel sm final.key (∑ i ∈ Finset.range count, lutBranchTarget (q := Q0) (p := (qd i0)) μ sp fam i)
        final.row (count * lutBranchBound777 i0 B Bh) ∧
      final.switch = fun k j ↦ fam k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q0 (R i0) 6) ↦ ∀ hk : k ≤ 3,
      Rel sm s.key (∑ i ∈ Finset.range k, lutBranchTarget (q := Q0) (p := (qd i0)) μ sp fam i) s.row
        (k * lutBranchBound777 i0 B Bh) ∧
        s.switch = fun k' j ↦ fam k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨?_, ?_⟩
    · rw [Finset.range_zero, Finset.sum_empty, zero_mul]
      have hrow : (![0, 0] : Fin 2 → ExactMatrix Q0 N 1 6) = 0 := by
        funext j
        fin_cases j <;> rfl
      simp only [LoopState.key, LoopState.row, hrow]
      exact Rel.zero
    · funext k j
      fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := lutStep777 hsm hg' hg henc hhelp (by omega) hsw hrel hbody
    refine ⟨?_, hsw'⟩
    rw [Finset.sum_range_succ, lutBranchTarget, dif_pos (by omega), add_mul, one_mul]
    exact hrel'

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem lut777 {i0 : Nat} {model : HashModel} {key : ByteArray} {params} {o : Nat} [Fact (i0 < 8)]
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix (R i0) N 1 6} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6} {outputs : Blocks (R i0) 1}
    (h : W_Stage_online_eval.parallel_parallel_generatedRoot_290_779 Backend.backend model params i0 o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11, ()) outputs)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {z : Nat}
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {B : Nat}
    (hinput : Rel sm ![![A00, A01], ![A10, A11]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * Xe (n := N) (2 * z) * unitPair sp j) • g)
      ![c0, c1] B)
    {Bh : Nat} (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    (ho : o < 160) {gq : ExactMatrix (R i0) N 1 6} (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq) {inverse : Nat}
    (hinverse : (inverse : ExactPoly (R i0) N) * ((4 : Nat) : ExactPoly (R i0) N) = 1) {BB : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient (R i0) inverse (fun k ↦ reducePoly (R i0) N (smudgeTable k)) (sigma i).toNat *
          reducePoly (R i0) N ((1 : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix (R i0) N 1 6 (e j)) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly (R i0) N (Xe (n := N) (o / 20)) • reducePoly (R i0) N
        (((smudgeWeight777 i0 ⟨o % 20, Nat.mod_lt _ (by inst_num)⟩ : Int) : ErrorPoly N) *
          ((1 : ErrorPoly N) * ((((-1 : Int) ^ (z / 4) : Int) : ErrorPoly N) *
            smudgeTable ⟨z % 4, Nat.mod_lt _ (by inst_num)⟩)) * unitPair sp j) •
        (1 : ExactMatrix (R i0) N 1 1))
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      (6 * ((3 * lutBranchBound777 i0 B Bh + ((6 + 6) * N * Bh * 16384 + (6 + 6) * N * B * 16384 +
        (N * 1 * ((qd i0) / 2) + (qd i0) / 2)) + (N + 2) * ((qd i0) / 2)) / (qd i0) + 4 * (N * 1 * BB)) * (32768 / 2)) := by
  obtain ⟨L, hL, hLx, -, hLd⟩ := layoutR i0 hi0
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_779.body at h
  extract_lets -merge _params _backend w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_73_concat_1 w_73_concat_2 w_73_concat_3 w_73_concat_4 w_73_concat_5 w_73_concat_6 w_73_concat_7 w_73_concat_8 w_73_concat_9 w_73_concat_10 w_73_concat_11 w_73_concat_12 w_73_concat_13 w_73_concat_14 w_73_concat_15 w_73_concat_16 w_73_concat_17 w_73_concat_18 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 at h
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_779.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_779.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_779.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_779.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_779.constraints_4, W_Stage_online_eval.parallel_parallel_generatedRoot_290_779.constraints_5] at h
  obtain ⟨h2_0, h4_0, h6_0, h8_0, -, h18_9, h19_0, -, -, h21_0, h22_0, h21_0_1, h23_0, h24_0, h25_0, h26_0, h25_0_1, h27_0, h28_0, -, -, h29_0, -, -, h30_0, -, -, h31_0, h30_0_1, h32_0, -, -, h33_0, -, -, h34_0, -, -, h35_0, h34_0_1, h36_0, h37_0, h38_0, h39_0, h40_0, -, -, -, -, -, -, h42_0, h43_0, -, -, -, -, -, -, h44_0, h45_0, h46_0, h47_0, h48_0, -, -, -, -, -, -, h50_0, h52_0, -, -, h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, -, -, -, -, -, -, -, -, h74_0, h75_0, h76_0, -, h80_0, h81_0, h82_0, -, -, -, -, -, -, h84_0, h85_0, -, -, -, -, -, -, h86_0, h87_0, h88_0, h89_0, h90_0, -, -, -, -, -, -, h92_0, h94_0, -, -, -, -, -, -, h97_0, h99_0, -, -, -, -, -, -, h102_0, h104_0, -, -, -, -, -, -, h107_0, -, -, -, -, -, -, h108_0, h111_0, -, -, -, -, -, -, h114_0, -, -, -, -, -, -, h115_0, h118_0, hout⟩ := h
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  rw [hwa, hwb, hwc, hwd] at h18_9
  rw [show Int.toNat 3 = 3 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := lutLoop777 (fam := fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j)
    (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]])
    (final := (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) hsm hg' hg hinput hhelp (count := 3) (by decide) h18_9
  obtain ⟨_, hw19⟩ := familyGet_eq (k := 3) h19_0
  obtain ⟨_, hw24⟩ := familyGet_eq (k := 3) h24_0
  rw [hw19] at h21_0_1
  rw [hw24] at h25_0_1
  have hA0 : w_21_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h21_0
  have hA1 : w_25_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h25_0
  have hA2 : w_22_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h22_0
  have hA3 : w_26_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h26_0
  rw [hA0] at h21_0_1
  rw [hA2] at h23_0
  rw [hA1] at h25_0_1
  rw [hA3] at h27_0
  have hhelp3 := hhelp 3
  have hfin := lut_round (q := Q0) (q' := (R i0)) (p := (qd i0)) (m := 6) (m' := 6) (q0_eqR i0 hi0) (by inst_num)
    (by inst_num) (by inst_num) L hL hLx hsm hg' hg hinput hloop hloopSwitch hhelp3
    h29_0 h30_0 h31_0 h33_0 h34_0 h35_0 h30_0_1 h32_0 h34_0_1 h36_0 h37_0
    (Xb := ![w_38_0, w_80_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h80_0])
    (D' := ![w_39_0, w_81_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h81_0])
    (D := ![w_40_0, w_82_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h82_0])
    (M := ![![w_42_0, w_84_0], ![w_44_0, w_86_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h84_0])
    (fun j ↦ by fin_cases j; exacts [h44_0, h86_0])
    (Asw := ![![w_43_0, w_85_0], ![w_45_0, w_87_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h43_0, h85_0, h45_0, h87_0])
    (Z := ![w_46_0, w_88_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h88_0])
    (E' := ![w_47_0, w_89_0]) (fun j ↦ by fin_cases j; exacts [h47_0, h89_0])
    (E := ![w_48_0, w_90_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h90_0])
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (Y := ![w_107_0, w_114_0]) (X := ![w_108_0, w_115_0])
    (K := ![![w_50_0, w_92_0], ![w_97_0, w_102_0]])
    (fun j ↦ by fin_cases j; exacts [h107_0, h114_0])
    (fun j ↦ by fin_cases j; exacts [h108_0, h115_0])
    (fun j ↦ by fin_cases j; exacts [h50_0, h92_0])
    (fun j ↦ by fin_cases j; exacts [h97_0, h102_0])
    (T := smudgeTable) (Δ' := (1 : ErrorPoly N)) hinverse
    (fun i ↦ hswitch i) (K' := ![![w_52_0, w_94_0], ![w_99_0, w_104_0]])
    (R' := ![w_111_0, w_118_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h52_0, h94_0, h99_0, h104_0])
    (fun j ↦ by fin_cases j; exacts [h111_0, h118_0])
  -- The weight `w_(o mod 20)` of the member's digit.
  simp only [concatColumns] at h54_0 h55_0 h56_0 h57_0 h58_0 h59_0 h60_0 h61_0 h62_0 h63_0 h64_0 h65_0 h66_0 h67_0 h68_0 h69_0 h70_0 h71_0 h73_0
  have hrow : ∀ c : Fin 20, w_73_0 0 c = (matrixPolynomial [smudgeWeight777 i0 c] : ExactMatrix (R i0) N 1 1) 0 0 := by
    intro c
    fin_cases c <;> simp [h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, w_53_0, w_54_0, w_55_0, w_56_0, w_57_0, w_58_0, w_59_0, w_60_0, w_61_0, w_62_0, w_63_0, w_64_0, w_65_0, w_66_0, w_67_0, w_68_0, w_69_0, w_70_0, w_71_0, w_72_0, smudgeWeight777]
  have hd : Int.fmod (Int.ofNat o) 20 = ((o % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by inst_num)]; rfl
  rw [hd] at h74_0
  have hK : w_74_0 = matrixPolynomial [smudgeWeight777 i0 ⟨o % 20, Nat.mod_lt _ (by inst_num)⟩] := by
    funext a b
    obtain rfl : a = 0 := Subsingleton.elim _ _
    obtain rfl : b = 0 := Subsingleton.elim _ _
    rw [slice_entry h74_0 (by simp; omega)]
    exact hrow _
  have hfin2 := hfin.finalizeCoefficient (by inst_num) hN L hL hLx hgq _ hK h75_0 h76_0
  have hpl := rel_monomial hfin2 (k := o / 20) (by simp only [N]; omega)
  have hq : w_78_0 = ((o / 20 : Nat) : Int) := by
    simp only [w_78_0]
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by inst_num)]; rfl
  rw [hout]
  convert hpl using 1
  all_goals simp only [w_79_0, w_96_0, w_101_0, w_106_0, w_113_0, w_120_0, w_77_0, w_95_0, w_100_0, w_105_0, w_112_0, w_119_0, hq]
  all_goals first
    | (funext a j; fin_cases a <;> fin_cases j <;> rfl)
    | (funext j; fin_cases j <;> rfl)

/-- The digit weights of the smudge LUT `831`. -/
def smudgeWeight831 (i0 : Nat) : Fin 20 → Int := ![(halfZ (i0 : Int)), 2, 8, 32, 128, 512, 2048, 8192, 32768, 131072, 524288, 2097152, 8388608, 33554432, 134217728, 536870912, 2147483648, 8589934592, 34359738368, 137438953472]

/-- The error bound added by one branch of the LUT `831`. -/
abbrev lutBranchBound831 (i0 : Nat) (B Bh : Nat) : Nat :=
  (6 + 6) * N * Bh * 16384 + (6 + 6) * N * B * 16384 + (N * 1 * ((qd i0) / 2) + (qd i0) / 2)

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- One iteration of the LUT loop `831` adds branch `i` to the accumulated relation and carries
the switching ciphertext of branch `i + 1`. -/
theorem lutStep831 {i0 : Nat} {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    [Fact (i0 < 8)] (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q0 N 1 6} {A : Fin 2 → Fin 2 → ExactMatrix Q0 N 1 6} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix (R i0) N 1 6}
    {i : Nat} (hi : i < 3) {state next : LoopState Q0 (R i0) 6}
    (hswitch : state.switch = fun k j ↦ fam k j ⟨i, by omega⟩)
    {T : Fin 2 → ExactMatrix Q0 N 1 6} {Bs : Nat} (hstate : Rel sm state.key T state.row Bs)
    {params} {o : Nat}
    (hbody : W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_833_18 Backend.backend model params i0 o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0, c 1, A 0 1, A 1 1, fam 0 0,
        fam 0 1, fam 1 0, fam 1 1, ()) next) :
    Rel sm next.key (T + branchTarget (q := Q0) (p := (qd i0)) μ sp fam ⟨i, by omega⟩) next.row
        (Bs + lutBranchBound831 i0 B Bh) ∧
      next.switch = fun k j ↦ fam k j ⟨i + 1, by omega⟩ := by
  obtain ⟨L, hL, hLx, -, hLd⟩ := layoutR i0 hi0
  unfold W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_833_18 at hbody
  extract_lets -merge _params _backend w_0_0 w_48_0 w_67_0 w_70_0 w_73_0 w_76_0 w_30_0 w_49_0 w_31_0 w_50_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_24_0 w_26_0 w_79_0 w_83_0 w_85_0 w_87_0 w_2_0 w_5_0 w_80_0 w_81_0 at hbody
  obtain ⟨w_3_0, hbody⟩ := hbody
  obtain ⟨w_6_0, hbody⟩ := hbody
  obtain ⟨w_7_0, hbody⟩ := hbody
  obtain ⟨w_8_0, hbody⟩ := hbody
  obtain ⟨w_8_concat_1, hbody⟩ := hbody
  obtain ⟨w_10_0, hbody⟩ := hbody
  obtain ⟨w_11_0, hbody⟩ := hbody
  obtain ⟨w_12_0, hbody⟩ := hbody
  obtain ⟨w_13_0, hbody⟩ := hbody
  obtain ⟨w_13_concat_1, hbody⟩ := hbody
  obtain ⟨w_14_0, hbody⟩ := hbody
  obtain ⟨w_16_0, hbody⟩ := hbody
  obtain ⟨w_18_0, hbody⟩ := hbody
  obtain ⟨w_20_0, hbody⟩ := hbody
  obtain ⟨w_21_0, hbody⟩ := hbody
  obtain ⟨w_21_concat_1, hbody⟩ := hbody
  obtain ⟨w_23_0, hbody⟩ := hbody
  obtain ⟨w_25_0, hbody⟩ := hbody
  obtain ⟨w_27_0, hbody⟩ := hbody
  obtain ⟨w_28_0, hbody⟩ := hbody
  obtain ⟨w_28_concat_1, hbody⟩ := hbody
  obtain ⟨w_29_0, hbody⟩ := hbody
  obtain ⟨w_32_0, hbody⟩ := hbody
  obtain ⟨w_33_0, hbody⟩ := hbody
  obtain ⟨w_34_0, hbody⟩ := hbody
  extract_lets -merge w_35_0 at hbody
  obtain ⟨w_36_0, hbody⟩ := hbody
  obtain ⟨w_37_0, hbody⟩ := hbody
  obtain ⟨w_38_0, hbody⟩ := hbody
  obtain ⟨w_39_0, hbody⟩ := hbody
  obtain ⟨w_40_0, hbody⟩ := hbody
  obtain ⟨w_41_0, hbody⟩ := hbody
  obtain ⟨w_42_0, hbody⟩ := hbody
  extract_lets -merge w_43_0 at hbody
  obtain ⟨w_44_0, hbody⟩ := hbody
  obtain ⟨w_45_0, hbody⟩ := hbody
  extract_lets -merge w_46_0 w_47_0 at hbody
  obtain ⟨w_51_0, hbody⟩ := hbody
  obtain ⟨w_52_0, hbody⟩ := hbody
  obtain ⟨w_53_0, hbody⟩ := hbody
  extract_lets -merge w_54_0 at hbody
  obtain ⟨w_55_0, hbody⟩ := hbody
  obtain ⟨w_56_0, hbody⟩ := hbody
  obtain ⟨w_57_0, hbody⟩ := hbody
  obtain ⟨w_58_0, hbody⟩ := hbody
  obtain ⟨w_59_0, hbody⟩ := hbody
  obtain ⟨w_60_0, hbody⟩ := hbody
  obtain ⟨w_61_0, hbody⟩ := hbody
  extract_lets -merge w_62_0 at hbody
  obtain ⟨w_63_0, hbody⟩ := hbody
  obtain ⟨w_64_0, hbody⟩ := hbody
  extract_lets -merge w_65_0 w_66_0 at hbody
  obtain ⟨w_68_0, hbody⟩ := hbody
  extract_lets -merge w_69_0 at hbody
  obtain ⟨w_71_0, hbody⟩ := hbody
  extract_lets -merge w_72_0 at hbody
  obtain ⟨w_74_0, hbody⟩ := hbody
  extract_lets -merge w_75_0 at hbody
  obtain ⟨w_77_0, hbody⟩ := hbody
  extract_lets -merge w_78_0 at hbody
  obtain ⟨w_82_0, hbody⟩ := hbody
  obtain ⟨w_84_0, hbody⟩ := hbody
  obtain ⟨w_86_0, hbody⟩ := hbody
  obtain ⟨w_88_0, hbody⟩ := hbody
  simp only [W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_833_18.constraints_0, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_833_18.constraints_1, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_833_18.constraints_2, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_833_18.constraints_3, W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_833_18.constraints_4] at hbody
  obtain ⟨h2_0, h2_0_1, h3_0, -, -, h6_0, h7_0, h6_0_1, h8_0, h2_0_2, h2_0_3, h10_0, h11_0, h12_0, h11_0_1, h13_0, h14_0, -, -, h16_0, -, -, h18_0, -, -, h20_0, h18_0_1, h21_0, -, -, h23_0, -, -, h25_0, -, -, h27_0, h25_0_1, h28_0, h29_0, h32_0, h33_0, h34_0, -, -, -, -, -, -, h36_0, h37_0, -, -, -, -, -, -, h38_0, h39_0, h40_0, h41_0, h42_0, -, -, -, -, -, -, h44_0, -, -, -, -, -, -, h45_0, h51_0, h52_0, h53_0, -, -, -, -, -, -, h55_0, h56_0, -, -, -, -, -, -, h57_0, h58_0, h59_0, h60_0, h61_0, -, -, -, -, -, -, h63_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h77_0, h81_0, h81_0_1, h82_0, h81_0_2, h81_0_3, h84_0, h81_0_4, h81_0_5, h86_0, h81_0_6, h81_0_7, h88_0, hout⟩ := hbody
  have hidx : w_2_0 = Int.ofNat i := rfl
  have hnext : w_81_0 = Int.ofNat (i + 1) := by
    simp only [w_81_0, w_80_0, w_2_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h3_0 h10_0
  rw [hnext] at h82_0 h84_0 h86_0 h88_0
  obtain ⟨_, hw3⟩ := familyGet_eq h3_0
  obtain ⟨_, hw10⟩ := familyGet_eq h10_0
  obtain ⟨_, hs0⟩ := familyGet_eq h82_0
  obtain ⟨_, hs1⟩ := familyGet_eq h84_0
  obtain ⟨_, hs2⟩ := familyGet_eq h86_0
  obtain ⟨_, hs3⟩ := familyGet_eq h88_0
  subst hw3 hw10
  have hhelp' := hhelp ⟨i, by omega⟩
  have hAd : lutA model key Q0 6 2 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  simp only [hAd] at hhelp'
  have hbranch := lut_branch (n := N) (q := Q0) (q' := (R i0)) (p := (qd i0)) (m := 6) (m' := 6)
    (log := 3) (bound := 16384) (index := sigma i) hN hpow (q0_eqR i0 hi0) (by inst_num) (by inst_num)
    L hL hLx hg' hg hsm
    henc h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (bsw := state.switch) (Xb := ![w_32_0, w_51_0])
    (fun j ↦ by fin_cases j; exacts [h32_0, h51_0])
    (D' := ![w_33_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h33_0, h52_0])
    (D := ![w_34_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h34_0, h53_0])
    (M := ![![w_36_0, w_55_0], ![w_38_0, w_57_0]])
    (fun j ↦ by fin_cases j; exacts [h36_0, h55_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h57_0])
    (Asw := ![![w_37_0, w_56_0], ![w_39_0, w_58_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h37_0, h56_0, h39_0, h58_0])
    (Z := ![w_40_0, w_59_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h59_0])
    (E' := ![w_41_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h41_0, h60_0])
    (E := ![w_42_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h42_0, h61_0])
    (h := ![_, _]) (sigma_odd ⟨i, by omega⟩) hhelp' h6_0_1 h8_0 h11_0_1 h13_0 h14_0
    (Y := ![w_44_0, w_63_0]) (X := ![w_45_0, w_64_0])
    (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h44_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h45_0, h64_0])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  subst hout
  rw [hswitch] at hbranch
  refine ⟨?_, ?_⟩
  · convert hstate.add hbranch using 1
    · funext k j
      fin_cases k <;> fin_cases j <;> rfl
    · funext j
      fin_cases j <;> rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hs0, hs1, hs2, hs3]

/-- The generated loop `831` accumulates the first `count` branches and carries the switching
ciphertext of branch `count`. -/
theorem lutLoop831 {i0 : Nat} {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    [Fact (i0 < 8)] (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q0 N 1 6} {A : Fin 2 → Fin 2 → ExactMatrix Q0 N 1 6} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix (R i0) N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q0 (R i0) 6} {params} {o : Nat}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q0 (R i0) 6) ↦
      W_Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_833_18 Backend.backend model params i0 o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1,
          current.2.2.2.2.2.1, current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1,
          current.2.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0,
          c 1, A 0 1, A 1 1, fam 0 0, fam 0 1, fam 1 0, fam 1 1, ()) next) count
      (0, 0, 0, 0, 0, 0, fam 0 0 0, fam 0 1 0, fam 1 0 0, fam 1 1 0, ()) final) :
    Rel sm final.key (∑ i ∈ Finset.range count, lutBranchTarget (q := Q0) (p := (qd i0)) μ sp fam i)
        final.row (count * lutBranchBound831 i0 B Bh) ∧
      final.switch = fun k j ↦ fam k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q0 (R i0) 6) ↦ ∀ hk : k ≤ 3,
      Rel sm s.key (∑ i ∈ Finset.range k, lutBranchTarget (q := Q0) (p := (qd i0)) μ sp fam i) s.row
        (k * lutBranchBound831 i0 B Bh) ∧
        s.switch = fun k' j ↦ fam k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨?_, ?_⟩
    · rw [Finset.range_zero, Finset.sum_empty, zero_mul]
      have hrow : (![0, 0] : Fin 2 → ExactMatrix Q0 N 1 6) = 0 := by
        funext j
        fin_cases j <;> rfl
      simp only [LoopState.key, LoopState.row, hrow]
      exact Rel.zero
    · funext k j
      fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := lutStep831 hsm hg' hg henc hhelp (by omega) hsw hrel hbody
    refine ⟨?_, hsw'⟩
    rw [Finset.sum_range_succ, lutBranchTarget, dif_pos (by omega), add_mul, one_mul]
    exact hrel'

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem lut831 {i0 : Nat} {model : HashModel} {key : ByteArray} {params} {o : Nat} [Fact (i0 < 8)]
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix (R i0) N 1 6} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6} {outputs : Blocks (R i0) 1}
    (h : W_Stage_online_eval.parallel_parallel_generatedRoot_290_833 Backend.backend model params i0 o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11, ()) outputs)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {z : Nat}
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {B : Nat}
    (hinput : Rel sm ![![A00, A01], ![A10, A11]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * Xe (n := N) (2 * z) * unitPair sp j) • g)
      ![c0, c1] B)
    {Bh : Nat} (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    (ho : o < 160) {gq : ExactMatrix (R i0) N 1 6} (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq) {inverse : Nat}
    (hinverse : (inverse : ExactPoly (R i0) N) * ((4 : Nat) : ExactPoly (R i0) N) = 1) {BB : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient (R i0) inverse (fun k ↦ reducePoly (R i0) N (smudgeTable k)) (sigma i).toNat *
          reducePoly (R i0) N ((1 : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix (R i0) N 1 6 (e j)) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly (R i0) N (Xe (n := N) (o / 20)) • reducePoly (R i0) N
        (((smudgeWeight831 i0 ⟨o % 20, Nat.mod_lt _ (by inst_num)⟩ : Int) : ErrorPoly N) *
          ((1 : ErrorPoly N) * ((((-1 : Int) ^ (z / 4) : Int) : ErrorPoly N) *
            smudgeTable ⟨z % 4, Nat.mod_lt _ (by inst_num)⟩)) * unitPair sp j) •
        (1 : ExactMatrix (R i0) N 1 1))
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      (6 * ((3 * lutBranchBound831 i0 B Bh + ((6 + 6) * N * Bh * 16384 + (6 + 6) * N * B * 16384 +
        (N * 1 * ((qd i0) / 2) + (qd i0) / 2)) + (N + 2) * ((qd i0) / 2)) / (qd i0) + 4 * (N * 1 * BB)) * (32768 / 2)) := by
  obtain ⟨L, hL, hLx, -, hLd⟩ := layoutR i0 hi0
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_833.body at h
  extract_lets -merge _params _backend w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_73_concat_1 w_73_concat_2 w_73_concat_3 w_73_concat_4 w_73_concat_5 w_73_concat_6 w_73_concat_7 w_73_concat_8 w_73_concat_9 w_73_concat_10 w_73_concat_11 w_73_concat_12 w_73_concat_13 w_73_concat_14 w_73_concat_15 w_73_concat_16 w_73_concat_17 w_73_concat_18 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 at h
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_833.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_833.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_833.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_833.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_833.constraints_4, W_Stage_online_eval.parallel_parallel_generatedRoot_290_833.constraints_5] at h
  obtain ⟨h2_0, h4_0, h6_0, h8_0, -, h18_9, h19_0, -, -, h21_0, h22_0, h21_0_1, h23_0, h24_0, h25_0, h26_0, h25_0_1, h27_0, h28_0, -, -, h29_0, -, -, h30_0, -, -, h31_0, h30_0_1, h32_0, -, -, h33_0, -, -, h34_0, -, -, h35_0, h34_0_1, h36_0, h37_0, h38_0, h39_0, h40_0, -, -, -, -, -, -, h42_0, h43_0, -, -, -, -, -, -, h44_0, h45_0, h46_0, h47_0, h48_0, -, -, -, -, -, -, h50_0, h52_0, -, -, h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, -, -, -, -, -, -, -, -, h74_0, h75_0, h76_0, -, h80_0, h81_0, h82_0, -, -, -, -, -, -, h84_0, h85_0, -, -, -, -, -, -, h86_0, h87_0, h88_0, h89_0, h90_0, -, -, -, -, -, -, h92_0, h94_0, -, -, -, -, -, -, h97_0, h99_0, -, -, -, -, -, -, h102_0, h104_0, -, -, -, -, -, -, h107_0, -, -, -, -, -, -, h108_0, h111_0, -, -, -, -, -, -, h114_0, -, -, -, -, -, -, h115_0, h118_0, hout⟩ := h
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  rw [hwa, hwb, hwc, hwd] at h18_9
  rw [show Int.toNat 3 = 3 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := lutLoop831 (fam := fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j)
    (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]])
    (final := (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) hsm hg' hg hinput hhelp (count := 3) (by decide) h18_9
  obtain ⟨_, hw19⟩ := familyGet_eq (k := 3) h19_0
  obtain ⟨_, hw24⟩ := familyGet_eq (k := 3) h24_0
  rw [hw19] at h21_0_1
  rw [hw24] at h25_0_1
  have hA0 : w_21_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h21_0
  have hA1 : w_25_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h25_0
  have hA2 : w_22_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h22_0
  have hA3 : w_26_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h26_0
  rw [hA0] at h21_0_1
  rw [hA2] at h23_0
  rw [hA1] at h25_0_1
  rw [hA3] at h27_0
  have hhelp3 := hhelp 3
  have hfin := lut_round (q := Q0) (q' := (R i0)) (p := (qd i0)) (m := 6) (m' := 6) (q0_eqR i0 hi0) (by inst_num)
    (by inst_num) (by inst_num) L hL hLx hsm hg' hg hinput hloop hloopSwitch hhelp3
    h29_0 h30_0 h31_0 h33_0 h34_0 h35_0 h30_0_1 h32_0 h34_0_1 h36_0 h37_0
    (Xb := ![w_38_0, w_80_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h80_0])
    (D' := ![w_39_0, w_81_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h81_0])
    (D := ![w_40_0, w_82_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h82_0])
    (M := ![![w_42_0, w_84_0], ![w_44_0, w_86_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h84_0])
    (fun j ↦ by fin_cases j; exacts [h44_0, h86_0])
    (Asw := ![![w_43_0, w_85_0], ![w_45_0, w_87_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h43_0, h85_0, h45_0, h87_0])
    (Z := ![w_46_0, w_88_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h88_0])
    (E' := ![w_47_0, w_89_0]) (fun j ↦ by fin_cases j; exacts [h47_0, h89_0])
    (E := ![w_48_0, w_90_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h90_0])
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (Y := ![w_107_0, w_114_0]) (X := ![w_108_0, w_115_0])
    (K := ![![w_50_0, w_92_0], ![w_97_0, w_102_0]])
    (fun j ↦ by fin_cases j; exacts [h107_0, h114_0])
    (fun j ↦ by fin_cases j; exacts [h108_0, h115_0])
    (fun j ↦ by fin_cases j; exacts [h50_0, h92_0])
    (fun j ↦ by fin_cases j; exacts [h97_0, h102_0])
    (T := smudgeTable) (Δ' := (1 : ErrorPoly N)) hinverse
    (fun i ↦ hswitch i) (K' := ![![w_52_0, w_94_0], ![w_99_0, w_104_0]])
    (R' := ![w_111_0, w_118_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h52_0, h94_0, h99_0, h104_0])
    (fun j ↦ by fin_cases j; exacts [h111_0, h118_0])
  -- The weight `w_(o mod 20)` of the member's digit.
  simp only [concatColumns] at h54_0 h55_0 h56_0 h57_0 h58_0 h59_0 h60_0 h61_0 h62_0 h63_0 h64_0 h65_0 h66_0 h67_0 h68_0 h69_0 h70_0 h71_0 h73_0
  have hrow : ∀ c : Fin 20, w_73_0 0 c = (matrixPolynomial [smudgeWeight831 i0 c] : ExactMatrix (R i0) N 1 1) 0 0 := by
    intro c
    fin_cases c <;> simp [h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, w_53_0, w_54_0, w_55_0, w_56_0, w_57_0, w_58_0, w_59_0, w_60_0, w_61_0, w_62_0, w_63_0, w_64_0, w_65_0, w_66_0, w_67_0, w_68_0, w_69_0, w_70_0, w_71_0, w_72_0, smudgeWeight831]
  have hd : Int.fmod (Int.ofNat o) 20 = ((o % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by inst_num)]; rfl
  rw [hd] at h74_0
  have hK : w_74_0 = matrixPolynomial [smudgeWeight831 i0 ⟨o % 20, Nat.mod_lt _ (by inst_num)⟩] := by
    funext a b
    obtain rfl : a = 0 := Subsingleton.elim _ _
    obtain rfl : b = 0 := Subsingleton.elim _ _
    rw [slice_entry h74_0 (by simp; omega)]
    exact hrow _
  have hfin2 := hfin.finalizeCoefficient (by inst_num) hN L hL hLx hgq _ hK h75_0 h76_0
  have hpl := rel_monomial hfin2 (k := o / 20) (by simp only [N]; omega)
  have hq : w_78_0 = ((o / 20 : Nat) : Int) := by
    simp only [w_78_0]
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by inst_num)]; rfl
  rw [hout]
  convert hpl using 1
  all_goals simp only [w_79_0, w_96_0, w_101_0, w_106_0, w_113_0, w_120_0, w_77_0, w_95_0, w_100_0, w_105_0, w_112_0, w_119_0, hq]
  all_goals first
    | (funext a j; fin_cases a <;> fin_cases j <;> rfl)
    | (funext j; fin_cases j <;> rfl)

end RefreshTopProof.Part_I
