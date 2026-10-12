import TopLut
import TopInst

/-! The PRF readout and sampling LUTs of the mask and the fresh noise (written by
`generator/gen_online_luts.py`). -/

namespace RefreshTopProof.Part_I

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

/-- The readout table of the PRF: `X^(2 (2 f(k) - 1))`, `f = (1, 1, 2, 2)`. -/
noncomputable def readoutTable (k : Fin 4) : ErrorPoly N := Xe (2 * ![1, 1, 3, 3] k)

/-- The error bound added by one branch of the LUT `121`. -/
abbrev lutBranchBound121 (B Bh : Nat) : Nat :=
  (8 + 8) * N * Bh * 16384 + (8 + 8) * N * B * 16384 + (N * 1 * (p4 / 2) + p4 / 2)

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- One iteration of the LUT loop `121` adds branch `i` to the accumulated relation and carries
the switching ciphertext of branch `i + 1`. -/
theorem lutStep121 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q5 N 1 8} {A : Fin 2 → Fin 2 → ExactMatrix Q5 N 1 8} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {h0 h1 : Fin 4 → ExactMatrix Q5 N 1 8} {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q5 8 2 (sigma i))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix Q0 N 1 6}
    {i : Nat} (hi : i < 3) {state next : LoopState Q5 Q0 6}
    (hswitch : state.switch = fun k j ↦ fam k j ⟨i, by omega⟩)
    {T : Fin 2 → ExactMatrix Q5 N 1 6} {Bs : Nat} (hstate : Rel sm state.key T state.row Bs)
    {params} {o : Nat}
    (hbody : Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_121_18 Backend.backend model params i0 o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0, c 1, A 0 1, A 1 1, fam 0 0,
        fam 0 1, fam 1 0, fam 1 1, ()) next) :
    Rel sm next.key (T + branchTarget (q := Q5) (p := p4) μ sp fam ⟨i, by omega⟩) next.row
        (Bs + lutBranchBound121 B Bh) ∧
      next.switch = fun k j ↦ fam k j ⟨i + 1, by omega⟩ := by
  unfold Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_121_18 at hbody
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
  simp only [Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_121_18.constraints_0, Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_121_18.constraints_1, Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_121_18.constraints_2, Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_121_18.constraints_3, Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_121_18.constraints_4] at hbody
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
  have hAd : lutA model key Q5 8 2 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  simp only [hAd] at hhelp'
  have hbranch := lut_branch (n := N) (q := Q5) (q' := Q0) (p := p4) (m := 8) (m' := 6)
    (log := 3) (bound := 16384) (index := sigma i) hN hpow q5_eq (by inst_num) (by inst_num)
    Backend.layout5 layoutQ0 rfl hg' hg hsm
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

/-- The generated loop `121` accumulates the first `count` branches and carries the switching
ciphertext of branch `count`. -/
theorem lutLoop121 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q5 N 1 8} {A : Fin 2 → Fin 2 → ExactMatrix Q5 N 1 8} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {h0 h1 : Fin 4 → ExactMatrix Q5 N 1 8} {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q5 8 2 (sigma i))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix Q0 N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q5 Q0 6} {params} {o : Nat}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q5 Q0 6) ↦
      Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_121_18 Backend.backend model params i0 o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1,
          current.2.2.2.2.2.1, current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1,
          current.2.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0,
          c 1, A 0 1, A 1 1, fam 0 0, fam 0 1, fam 1 0, fam 1 1, ()) next) count
      (0, 0, 0, 0, 0, 0, fam 0 0 0, fam 0 1 0, fam 1 0 0, fam 1 1 0, ()) final) :
    Rel sm final.key (∑ i ∈ Finset.range count, lutBranchTarget (q := Q5) (p := p4) μ sp fam i)
        final.row (count * lutBranchBound121 B Bh) ∧
      final.switch = fun k j ↦ fam k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q5 Q0 6) ↦ ∀ hk : k ≤ 3,
      Rel sm s.key (∑ i ∈ Finset.range k, lutBranchTarget (q := Q5) (p := p4) μ sp fam i) s.row
        (k * lutBranchBound121 B Bh) ∧
        s.switch = fun k' j ↦ fam k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨?_, ?_⟩
    · rw [Finset.range_zero, Finset.sum_empty, zero_mul]
      have hrow : (![0, 0] : Fin 2 → ExactMatrix Q5 N 1 6) = 0 := by
        funext j
        fin_cases j <;> rfl
      simp only [LoopState.key, LoopState.row, hrow]
      exact Rel.zero
    · funext k j
      fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := lutStep121 hsm hg' hg henc hhelp (by omega) hsw hrel hbody
    refine ⟨?_, hsw'⟩
    rw [Finset.sum_range_succ, lutBranchTarget, dif_pos (by omega), add_mul, one_mul]
    exact hrel'

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem lut121 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {key : ByteArray} {params} {o : Nat}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q0 N 1 6} {h0 h1 : Fin 4 → ExactMatrix Q5 N 1 8}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q5 N 1 8} {outputs : Blocks Q0 6}
    (h : Stage_online_eval.parallel_parallel_generatedRoot_290_121 Backend.backend model params i0 o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11, ()) outputs)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {z : Nat}
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g)
    {B : Nat}
    (hinput : Rel sm ![![A00, A01], ![A10, A11]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * z) * unitPair sp j) • g)
      ![c0, c1] B)
    {Bh : Nat} (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q5 8 2 (sigma i))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {gq : ExactMatrix Q0 N 1 6} {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q0 N) * ((4 : Nat) : ExactPoly Q0 N) = 1) {BB : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly Q0 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient Q0 inverse (fun k ↦ reducePoly Q0 N (readoutTable k)) (sigma i).toNat *
          reducePoly Q0 N (((((qd i0) : Nat) : Int) : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix Q0 N 1 6 (e j)) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q0 N (((((qd i0) : Nat) : Int) : ErrorPoly N) * ((((-1 : Int) ^ (z / 4) : Int) : ErrorPoly N) *
        readoutTable ⟨z % 4, Nat.mod_lt _ (by inst_num)⟩) * unitPair sp j) • gq)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((3 * lutBranchBound121 B Bh + ((8 + 8) * N * Bh * 16384 + (8 + 8) * N * B * 16384 +
        (N * 1 * (p4 / 2) + p4 / 2)) + (N + 2) * (p4 / 2)) / p4 + 4 * (N * 1 * BB)) := by
  obtain ⟨w, h⟩ := h
  unfold Stage_online_eval.parallel_parallel_generatedRoot_290_121.body at h
  extract_lets -merge _params _backend w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 at h
  simp only [Stage_online_eval.parallel_parallel_generatedRoot_290_121.constraints_0, Stage_online_eval.parallel_parallel_generatedRoot_290_121.constraints_1, Stage_online_eval.parallel_parallel_generatedRoot_290_121.constraints_2, Stage_online_eval.parallel_parallel_generatedRoot_290_121.constraints_3, Stage_online_eval.parallel_parallel_generatedRoot_290_121.constraints_4] at h
  obtain ⟨h2_0, h4_0, h6_0, h8_0, -, h18_9, h19_0, -, -, h21_0, h22_0, h21_0_1, h23_0, h24_0, h25_0, h26_0, h25_0_1, h27_0, h28_0, -, -, h29_0, -, -, h30_0, -, -, h31_0, h30_0_1, h32_0, -, -, h33_0, -, -, h34_0, -, -, h35_0, h34_0_1, h36_0, h37_0, h38_0, h39_0, h40_0, -, -, -, -, -, -, h42_0, h43_0, -, -, -, -, -, -, h44_0, h45_0, h46_0, h47_0, h48_0, -, -, -, -, -, -, h50_0, h52_0, h53_0, h54_0, h55_0, -, -, -, -, -, -, h57_0, h58_0, -, -, -, -, -, -, h59_0, h60_0, h61_0, h62_0, h63_0, -, -, -, -, -, -, h65_0, h67_0, -, -, -, -, -, -, h68_0, h70_0, -, -, -, -, -, -, h71_0, h73_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h75_0, h78_0, -, -, -, -, -, -, h79_0, -, -, -, -, -, -, h80_0, h83_0, hout⟩ := h
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  rw [hwa, hwb, hwc, hwd] at h18_9
  rw [show Int.toNat 3 = 3 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := lutLoop121 (fam := fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j)
    (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]])
    (final := (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) hsm hg' hg hinput hhelp (count := 3) (by decide) h18_9
  obtain ⟨_, hw19⟩ := familyGet_eq (k := 3) h19_0
  obtain ⟨_, hw24⟩ := familyGet_eq (k := 3) h24_0
  rw [hw19] at h21_0_1
  rw [hw24] at h25_0_1
  have hA0 : w_21_0 = lutA model key Q5 8 2 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h21_0
  have hA1 : w_25_0 = lutA model key Q5 8 2 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h25_0
  have hA2 : w_22_0 = lutA model key Q5 8 2 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h22_0
  have hA3 : w_26_0 = lutA model key Q5 8 2 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h26_0
  rw [hA0] at h21_0_1
  rw [hA2] at h23_0
  rw [hA1] at h25_0_1
  rw [hA3] at h27_0
  have hhelp3 := hhelp 3
  have hfin := lut_round (q := Q5) (q' := Q0) (p := p4) (m := 8) (m' := 6) q5_eq (by inst_num)
    (by inst_num) (by inst_num) Backend.layout5 layoutQ0 rfl hsm hg' hg hinput hloop hloopSwitch hhelp3
    h29_0 h30_0 h31_0 h33_0 h34_0 h35_0 h30_0_1 h32_0 h34_0_1 h36_0 h37_0
    (Xb := ![w_38_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h53_0])
    (D' := ![w_39_0, w_54_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h54_0])
    (D := ![w_40_0, w_55_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h55_0])
    (M := ![![w_42_0, w_57_0], ![w_44_0, w_59_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h57_0])
    (fun j ↦ by fin_cases j; exacts [h44_0, h59_0])
    (Asw := ![![w_43_0, w_58_0], ![w_45_0, w_60_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h43_0, h58_0, h45_0, h60_0])
    (Z := ![w_46_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h61_0])
    (E' := ![w_47_0, w_62_0]) (fun j ↦ by fin_cases j; exacts [h47_0, h62_0])
    (E := ![w_48_0, w_63_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h63_0])
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (Y := ![w_74_0, w_79_0]) (X := ![w_75_0, w_80_0])
    (K := ![![w_50_0, w_65_0], ![w_68_0, w_71_0]])
    (fun j ↦ by fin_cases j; exacts [h74_0, h79_0])
    (fun j ↦ by fin_cases j; exacts [h75_0, h80_0])
    (fun j ↦ by fin_cases j; exacts [h50_0, h65_0])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0])
    (T := readoutTable) (Δ' := ((((qd i0) : Nat) : Int) : ErrorPoly N)) hinverse
    (fun i ↦ hswitch i) (K' := ![![w_52_0, w_67_0], ![w_70_0, w_73_0]])
    (R' := ![w_78_0, w_83_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h52_0, h67_0, h70_0, h73_0])
    (fun j ↦ by fin_cases j; exacts [h78_0, h83_0])
  rw [hout]
  convert hfin using 1
  all_goals first
    | rfl
    | (funext a j; fin_cases a <;> fin_cases j <;> rfl)
    | (funext j; fin_cases j <;> rfl)

/-- The error bound added by one branch of the LUT `312`. -/
abbrev lutBranchBound312 (i0 : Nat) (B Bh : Nat) : Nat :=
  (6 + 6) * N * Bh * 16384 + (6 + 6) * N * B * 16384 + (N * 1 * ((qd i0) / 2) + (qd i0) / 2)

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- One iteration of the LUT loop `312` adds branch `i` to the accumulated relation and carries
the switching ciphertext of branch `i + 1`. -/
theorem lutStep312 {i0 : Nat} {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    [Fact (i0 < 8)] (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q0 N 1 6} {A : Fin 2 → Fin 2 → ExactMatrix Q0 N 1 6} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 3 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix (R i0) N 1 6}
    {i : Nat} (hi : i < 3) {state next : LoopState Q0 (R i0) 6}
    (hswitch : state.switch = fun k j ↦ fam k j ⟨i, by omega⟩)
    {T : Fin 2 → ExactMatrix Q0 N 1 6} {Bs : Nat} (hstate : Rel sm state.key T state.row Bs)
    {params} {o : Nat}
    (hbody : Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_312_18 Backend.backend model params i0 o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0, c 1, A 0 1, A 1 1, fam 0 0,
        fam 0 1, fam 1 0, fam 1 1, ()) next) :
    Rel sm next.key (T + branchTarget (q := Q0) (p := (qd i0)) μ sp fam ⟨i, by omega⟩) next.row
        (Bs + lutBranchBound312 i0 B Bh) ∧
      next.switch = fun k j ↦ fam k j ⟨i + 1, by omega⟩ := by
  obtain ⟨L, hL, hLx, -, hLd⟩ := layoutR i0 hi0
  unfold Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_312_18 at hbody
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
  simp only [Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_312_18.constraints_0, Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_312_18.constraints_1, Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_312_18.constraints_2, Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_312_18.constraints_3, Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_312_18.constraints_4] at hbody
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
  have hAd : lutA model key Q0 6 3 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
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

/-- The generated loop `312` accumulates the first `count` branches and carries the switching
ciphertext of branch `count`. -/
theorem lutLoop312 {i0 : Nat} {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    [Fact (i0 < 8)] (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q0 N 1 6} {A : Fin 2 → Fin 2 → ExactMatrix Q0 N 1 6} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 3 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix (R i0) N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q0 (R i0) 6} {params} {o : Nat}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q0 (R i0) 6) ↦
      Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_312_18 Backend.backend model params i0 o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1,
          current.2.2.2.2.2.1, current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1,
          current.2.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0,
          c 1, A 0 1, A 1 1, fam 0 0, fam 0 1, fam 1 0, fam 1 1, ()) next) count
      (0, 0, 0, 0, 0, 0, fam 0 0 0, fam 0 1 0, fam 1 0 0, fam 1 1 0, ()) final) :
    Rel sm final.key (∑ i ∈ Finset.range count, lutBranchTarget (q := Q0) (p := (qd i0)) μ sp fam i)
        final.row (count * lutBranchBound312 i0 B Bh) ∧
      final.switch = fun k j ↦ fam k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q0 (R i0) 6) ↦ ∀ hk : k ≤ 3,
      Rel sm s.key (∑ i ∈ Finset.range k, lutBranchTarget (q := Q0) (p := (qd i0)) μ sp fam i) s.row
        (k * lutBranchBound312 i0 B Bh) ∧
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
    obtain ⟨hrel', hsw'⟩ := lutStep312 hsm hg' hg henc hhelp (by omega) hsw hrel hbody
    refine ⟨?_, hsw'⟩
    rw [Finset.sum_range_succ, lutBranchTarget, dif_pos (by omega), add_mul, one_mul]
    exact hrel'

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem lut312 {i0 : Nat} {model : HashModel} {key : ByteArray} {params} {o : Nat} [Fact (i0 < 8)]
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix (R i0) N 1 6} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6} {off : Blocks Q0 6} {outputs : Blocks Q0 6}
    (h : Stage_online_eval.parallel_parallel_generatedRoot_290_312 Backend.backend model params i0 o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11, off.1, off.2.1, off.2.2.1, off.2.2.2.1, off.2.2.2.2.1, off.2.2.2.2.2.1, ()) outputs)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {z : Nat}
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {B : Nat}
    (hinput : Rel sm ![![A00, A01], ![A10, A11]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * Xe (n := N) (2 * z) * unitPair sp j) • g)
      ![c0, c1] B)
    {Bh : Nat} (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 3 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {toff : Fin 2 → ExactMatrix Q0 N 1 6} {Boff : Nat}
    (hoff : Rel sm ![![off.1, off.2.1], ![off.2.2.1, off.2.2.2.1]] toff
      ![off.2.2.2.2.1, off.2.2.2.2.2.1] Boff) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ (∑ i : Fin 4, scaleUpMatrix Q0 (qd i0)
        (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * z))) •
          (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i))) + toff j)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((3 * lutBranchBound312 i0 B Bh + ((6 + 6) * N * Bh * 16384 + (6 + 6) * N * B * 16384 +
        (N * 1 * ((qd i0) / 2) + (qd i0) / 2))) + Boff) := by
  obtain ⟨L, hL, hLx, -, hLd⟩ := layoutR i0 hi0
  obtain ⟨w, h⟩ := h
  unfold Stage_online_eval.parallel_parallel_generatedRoot_290_312.body at h
  extract_lets -merge _params _backend w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_52_0 w_68_0 w_72_0 w_76_0 w_82_0 w_88_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_89_0 at h
  simp only [Stage_online_eval.parallel_parallel_generatedRoot_290_312.constraints_0, Stage_online_eval.parallel_parallel_generatedRoot_290_312.constraints_1, Stage_online_eval.parallel_parallel_generatedRoot_290_312.constraints_2, Stage_online_eval.parallel_parallel_generatedRoot_290_312.constraints_3, Stage_online_eval.parallel_parallel_generatedRoot_290_312.constraints_4] at h
  obtain ⟨h2_0, h4_0, h6_0, h8_0, -, h18_9, h19_0, -, -, h21_0, h22_0, h21_0_1, h23_0, h24_0, h25_0, h26_0, h25_0_1, h27_0, h28_0, -, -, h29_0, -, -, h30_0, -, -, h31_0, h30_0_1, h32_0, -, -, h33_0, -, -, h34_0, -, -, h35_0, h34_0_1, h36_0, h37_0, h38_0, h39_0, h40_0, -, -, -, -, -, -, h42_0, h43_0, -, -, -, -, -, -, h44_0, h45_0, h46_0, h47_0, h48_0, -, -, -, -, -, -, h50_0, h54_0, h55_0, h56_0, -, -, -, -, -, -, h58_0, h59_0, -, -, -, -, -, -, h60_0, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h66_0, -, -, -, -, -, -, h70_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h78_0, -, -, -, -, -, -, h79_0, -, -, -, -, -, -, h84_0, -, -, -, -, -, -, h85_0, hout⟩ := h
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  rw [hwa, hwb, hwc, hwd] at h18_9
  rw [show Int.toNat 3 = 3 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := lutLoop312 (fam := fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j)
    (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]])
    (final := (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) hsm hg' hg hinput hhelp (count := 3) (by decide) h18_9
  obtain ⟨_, hw19⟩ := familyGet_eq (k := 3) h19_0
  obtain ⟨_, hw24⟩ := familyGet_eq (k := 3) h24_0
  rw [hw19] at h21_0_1
  rw [hw24] at h25_0_1
  have hA0 : w_21_0 = lutA model key Q0 6 3 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h21_0
  have hA1 : w_25_0 = lutA model key Q0 6 3 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h25_0
  have hA2 : w_22_0 = lutA model key Q0 6 3 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h22_0
  have hA3 : w_26_0 = lutA model key Q0 6 3 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h26_0
  rw [hA0] at h21_0_1
  rw [hA2] at h23_0
  rw [hA1] at h25_0_1
  rw [hA3] at h27_0
  have hhelp3 := hhelp 3
  have hsum := lut_branches (q := Q0) (q' := (R i0)) (p := (qd i0)) (m := 6) (m' := 6) (q0_eqR i0 hi0) (by inst_num)
    (by inst_num) (by inst_num) L hL hLx hsm hg' hg hinput hloop hloopSwitch hhelp3
    h29_0 h30_0 h31_0 h33_0 h34_0 h35_0 h30_0_1 h32_0 h34_0_1 h36_0 h37_0
    (Xb := ![w_38_0, w_54_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h54_0])
    (D' := ![w_39_0, w_55_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h55_0])
    (D := ![w_40_0, w_56_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h56_0])
    (M := ![![w_42_0, w_58_0], ![w_44_0, w_60_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h58_0])
    (fun j ↦ by fin_cases j; exacts [h44_0, h60_0])
    (Asw := ![![w_43_0, w_59_0], ![w_45_0, w_61_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h43_0, h59_0, h45_0, h61_0])
    (Z := ![w_46_0, w_62_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h62_0])
    (E' := ![w_47_0, w_63_0]) (fun j ↦ by fin_cases j; exacts [h47_0, h63_0])
    (E := ![w_48_0, w_64_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h64_0])
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (Y := ![w_78_0, w_84_0]) (X := ![w_79_0, w_85_0])
    (K := ![![w_50_0, w_66_0], ![w_70_0, w_74_0]])
    (fun j ↦ by fin_cases j; exacts [h78_0, h84_0])
    (fun j ↦ by fin_cases j; exacts [h79_0, h85_0])
    (fun j ↦ by fin_cases j; exacts [h50_0, h66_0])
    (fun j ↦ by fin_cases j; exacts [h70_0, h74_0])
  have htot := hsum.add hoff
  rw [hout]
  convert htot using 1
  all_goals first
    | rfl
    | (funext a j; fin_cases a <;> fin_cases j <;> rfl)
    | (funext j; fin_cases j <;> rfl)

/-- The error bound added by one branch of the LUT `392`. -/
abbrev lutBranchBound392 (B Bh : Nat) : Nat :=
  (8 + 8) * N * Bh * 16384 + (8 + 8) * N * B * 16384 + (N * 1 * (p4 / 2) + p4 / 2)

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- One iteration of the LUT loop `392` adds branch `i` to the accumulated relation and carries
the switching ciphertext of branch `i + 1`. -/
theorem lutStep392 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q5 N 1 8} {A : Fin 2 → Fin 2 → ExactMatrix Q5 N 1 8} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {h0 h1 : Fin 4 → ExactMatrix Q5 N 1 8} {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q5 8 2 (sigma i))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix Q0 N 1 6}
    {i : Nat} (hi : i < 3) {state next : LoopState Q5 Q0 6}
    (hswitch : state.switch = fun k j ↦ fam k j ⟨i, by omega⟩)
    {T : Fin 2 → ExactMatrix Q5 N 1 6} {Bs : Nat} (hstate : Rel sm state.key T state.row Bs)
    {params} {o : Nat}
    (hbody : Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_394_18 Backend.backend model params i0 o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0, c 1, A 0 1, A 1 1, fam 0 0,
        fam 0 1, fam 1 0, fam 1 1, ()) next) :
    Rel sm next.key (T + branchTarget (q := Q5) (p := p4) μ sp fam ⟨i, by omega⟩) next.row
        (Bs + lutBranchBound392 B Bh) ∧
      next.switch = fun k j ↦ fam k j ⟨i + 1, by omega⟩ := by
  unfold Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_394_18 at hbody
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
  simp only [Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_394_18.constraints_0, Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_394_18.constraints_1, Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_394_18.constraints_2, Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_394_18.constraints_3, Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_394_18.constraints_4] at hbody
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
  have hAd : lutA model key Q5 8 2 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  simp only [hAd] at hhelp'
  have hbranch := lut_branch (n := N) (q := Q5) (q' := Q0) (p := p4) (m := 8) (m' := 6)
    (log := 3) (bound := 16384) (index := sigma i) hN hpow q5_eq (by inst_num) (by inst_num)
    Backend.layout5 layoutQ0 rfl hg' hg hsm
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

/-- The generated loop `392` accumulates the first `count` branches and carries the switching
ciphertext of branch `count`. -/
theorem lutLoop392 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q5 N 1 8} {A : Fin 2 → Fin 2 → ExactMatrix Q5 N 1 8} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {h0 h1 : Fin 4 → ExactMatrix Q5 N 1 8} {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q5 8 2 (sigma i))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix Q0 N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q5 Q0 6} {params} {o : Nat}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q5 Q0 6) ↦
      Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_394_18 Backend.backend model params i0 o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1,
          current.2.2.2.2.2.1, current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1,
          current.2.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0,
          c 1, A 0 1, A 1 1, fam 0 0, fam 0 1, fam 1 0, fam 1 1, ()) next) count
      (0, 0, 0, 0, 0, 0, fam 0 0 0, fam 0 1 0, fam 1 0 0, fam 1 1 0, ()) final) :
    Rel sm final.key (∑ i ∈ Finset.range count, lutBranchTarget (q := Q5) (p := p4) μ sp fam i)
        final.row (count * lutBranchBound392 B Bh) ∧
      final.switch = fun k j ↦ fam k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q5 Q0 6) ↦ ∀ hk : k ≤ 3,
      Rel sm s.key (∑ i ∈ Finset.range k, lutBranchTarget (q := Q5) (p := p4) μ sp fam i) s.row
        (k * lutBranchBound392 B Bh) ∧
        s.switch = fun k' j ↦ fam k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨?_, ?_⟩
    · rw [Finset.range_zero, Finset.sum_empty, zero_mul]
      have hrow : (![0, 0] : Fin 2 → ExactMatrix Q5 N 1 6) = 0 := by
        funext j
        fin_cases j <;> rfl
      simp only [LoopState.key, LoopState.row, hrow]
      exact Rel.zero
    · funext k j
      fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := lutStep392 hsm hg' hg henc hhelp (by omega) hsw hrel hbody
    refine ⟨?_, hsw'⟩
    rw [Finset.sum_range_succ, lutBranchTarget, dif_pos (by omega), add_mul, one_mul]
    exact hrel'

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem lut392 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {key : ByteArray} {params} {o : Nat}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q0 N 1 6} {h0 h1 : Fin 4 → ExactMatrix Q5 N 1 8}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q5 N 1 8} {outputs : Blocks Q0 6}
    (h : Stage_online_eval.parallel_parallel_generatedRoot_290_394 Backend.backend model params i0 o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11, ()) outputs)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {z : Nat}
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g)
    {B : Nat}
    (hinput : Rel sm ![![A00, A01], ![A10, A11]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * z) * unitPair sp j) • g)
      ![c0, c1] B)
    {Bh : Nat} (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q5 8 2 (sigma i))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {gq : ExactMatrix Q0 N 1 6} {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q0 N) * ((4 : Nat) : ExactPoly Q0 N) = 1) {BB : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly Q0 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient Q0 inverse (fun k ↦ reducePoly Q0 N (readoutTable k)) (sigma i).toNat *
          reducePoly Q0 N (((((qd i0) : Nat) : Int) : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix Q0 N 1 6 (e j)) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly Q0 N (((((qd i0) : Nat) : Int) : ErrorPoly N) * ((((-1 : Int) ^ (z / 4) : Int) : ErrorPoly N) *
        readoutTable ⟨z % 4, Nat.mod_lt _ (by inst_num)⟩) * unitPair sp j) • gq)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((3 * lutBranchBound392 B Bh + ((8 + 8) * N * Bh * 16384 + (8 + 8) * N * B * 16384 +
        (N * 1 * (p4 / 2) + p4 / 2)) + (N + 2) * (p4 / 2)) / p4 + 4 * (N * 1 * BB)) := by
  obtain ⟨w, h⟩ := h
  unfold Stage_online_eval.parallel_parallel_generatedRoot_290_394.body at h
  extract_lets -merge _params _backend w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 at h
  simp only [Stage_online_eval.parallel_parallel_generatedRoot_290_394.constraints_0, Stage_online_eval.parallel_parallel_generatedRoot_290_394.constraints_1, Stage_online_eval.parallel_parallel_generatedRoot_290_394.constraints_2, Stage_online_eval.parallel_parallel_generatedRoot_290_394.constraints_3, Stage_online_eval.parallel_parallel_generatedRoot_290_394.constraints_4] at h
  obtain ⟨h2_0, h4_0, h6_0, h8_0, -, h18_9, h19_0, -, -, h21_0, h22_0, h21_0_1, h23_0, h24_0, h25_0, h26_0, h25_0_1, h27_0, h28_0, -, -, h29_0, -, -, h30_0, -, -, h31_0, h30_0_1, h32_0, -, -, h33_0, -, -, h34_0, -, -, h35_0, h34_0_1, h36_0, h37_0, h38_0, h39_0, h40_0, -, -, -, -, -, -, h42_0, h43_0, -, -, -, -, -, -, h44_0, h45_0, h46_0, h47_0, h48_0, -, -, -, -, -, -, h50_0, h52_0, h53_0, h54_0, h55_0, -, -, -, -, -, -, h57_0, h58_0, -, -, -, -, -, -, h59_0, h60_0, h61_0, h62_0, h63_0, -, -, -, -, -, -, h65_0, h67_0, -, -, -, -, -, -, h68_0, h70_0, -, -, -, -, -, -, h71_0, h73_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h75_0, h78_0, -, -, -, -, -, -, h79_0, -, -, -, -, -, -, h80_0, h83_0, hout⟩ := h
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  rw [hwa, hwb, hwc, hwd] at h18_9
  rw [show Int.toNat 3 = 3 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := lutLoop392 (fam := fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j)
    (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]])
    (final := (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) hsm hg' hg hinput hhelp (count := 3) (by decide) h18_9
  obtain ⟨_, hw19⟩ := familyGet_eq (k := 3) h19_0
  obtain ⟨_, hw24⟩ := familyGet_eq (k := 3) h24_0
  rw [hw19] at h21_0_1
  rw [hw24] at h25_0_1
  have hA0 : w_21_0 = lutA model key Q5 8 2 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h21_0
  have hA1 : w_25_0 = lutA model key Q5 8 2 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h25_0
  have hA2 : w_22_0 = lutA model key Q5 8 2 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h22_0
  have hA3 : w_26_0 = lutA model key Q5 8 2 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h26_0
  rw [hA0] at h21_0_1
  rw [hA2] at h23_0
  rw [hA1] at h25_0_1
  rw [hA3] at h27_0
  have hhelp3 := hhelp 3
  have hfin := lut_round (q := Q5) (q' := Q0) (p := p4) (m := 8) (m' := 6) q5_eq (by inst_num)
    (by inst_num) (by inst_num) Backend.layout5 layoutQ0 rfl hsm hg' hg hinput hloop hloopSwitch hhelp3
    h29_0 h30_0 h31_0 h33_0 h34_0 h35_0 h30_0_1 h32_0 h34_0_1 h36_0 h37_0
    (Xb := ![w_38_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h53_0])
    (D' := ![w_39_0, w_54_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h54_0])
    (D := ![w_40_0, w_55_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h55_0])
    (M := ![![w_42_0, w_57_0], ![w_44_0, w_59_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h57_0])
    (fun j ↦ by fin_cases j; exacts [h44_0, h59_0])
    (Asw := ![![w_43_0, w_58_0], ![w_45_0, w_60_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h43_0, h58_0, h45_0, h60_0])
    (Z := ![w_46_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h61_0])
    (E' := ![w_47_0, w_62_0]) (fun j ↦ by fin_cases j; exacts [h47_0, h62_0])
    (E := ![w_48_0, w_63_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h63_0])
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (Y := ![w_74_0, w_79_0]) (X := ![w_75_0, w_80_0])
    (K := ![![w_50_0, w_65_0], ![w_68_0, w_71_0]])
    (fun j ↦ by fin_cases j; exacts [h74_0, h79_0])
    (fun j ↦ by fin_cases j; exacts [h75_0, h80_0])
    (fun j ↦ by fin_cases j; exacts [h50_0, h65_0])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0])
    (T := readoutTable) (Δ' := ((((qd i0) : Nat) : Int) : ErrorPoly N)) hinverse
    (fun i ↦ hswitch i) (K' := ![![w_52_0, w_67_0], ![w_70_0, w_73_0]])
    (R' := ![w_78_0, w_83_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h52_0, h67_0, h70_0, h73_0])
    (fun j ↦ by fin_cases j; exacts [h78_0, h83_0])
  rw [hout]
  convert hfin using 1
  all_goals first
    | rfl
    | (funext a j; fin_cases a <;> fin_cases j <;> rfl)
    | (funext j; fin_cases j <;> rfl)

/-- The error bound added by one branch of the LUT `393`. -/
abbrev lutBranchBound393 (i0 : Nat) (B Bh : Nat) : Nat :=
  (6 + 6) * N * Bh * 16384 + (6 + 6) * N * B * 16384 + (N * 1 * ((qd i0) / 2) + (qd i0) / 2)

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- One iteration of the LUT loop `393` adds branch `i` to the accumulated relation and carries
the switching ciphertext of branch `i + 1`. -/
theorem lutStep393 {i0 : Nat} {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    [Fact (i0 < 8)] (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q0 N 1 6} {A : Fin 2 → Fin 2 → ExactMatrix Q0 N 1 6} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 3 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix (R i0) N 1 6}
    {i : Nat} (hi : i < 3) {state next : LoopState Q0 (R i0) 6}
    (hswitch : state.switch = fun k j ↦ fam k j ⟨i, by omega⟩)
    {T : Fin 2 → ExactMatrix Q0 N 1 6} {Bs : Nat} (hstate : Rel sm state.key T state.row Bs)
    {params} {o : Nat}
    (hbody : Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_395_18 Backend.backend model params i0 o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0, c 1, A 0 1, A 1 1, fam 0 0,
        fam 0 1, fam 1 0, fam 1 1, ()) next) :
    Rel sm next.key (T + branchTarget (q := Q0) (p := (qd i0)) μ sp fam ⟨i, by omega⟩) next.row
        (Bs + lutBranchBound393 i0 B Bh) ∧
      next.switch = fun k j ↦ fam k j ⟨i + 1, by omega⟩ := by
  obtain ⟨L, hL, hLx, -, hLd⟩ := layoutR i0 hi0
  unfold Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_395_18 at hbody
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
  simp only [Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_395_18.constraints_0, Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_395_18.constraints_1, Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_395_18.constraints_2, Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_395_18.constraints_3, Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_395_18.constraints_4] at hbody
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
  have hAd : lutA model key Q0 6 3 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
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

/-- The generated loop `393` accumulates the first `count` branches and carries the switching
ciphertext of branch `count`. -/
theorem lutLoop393 {i0 : Nat} {model : HashModel} {key : ByteArray} {sm sp μ : ErrorPoly N}
    [Fact (i0 < 8)] (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix Q0 N 1 6} {A : Fin 2 → Fin 2 → ExactMatrix Q0 N 1 6} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 3 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh)
    {fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix (R i0) N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q0 (R i0) 6} {params} {o : Nat}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q0 (R i0) 6) ↦
      Stage_online_eval.sequential_parallel_parallel_generatedRoot_290_395_18 Backend.backend model params i0 o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1,
          current.2.2.2.2.2.1, current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1,
          current.2.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c 0, A 0 0, A 1 0,
          c 1, A 0 1, A 1 1, fam 0 0, fam 0 1, fam 1 0, fam 1 1, ()) next) count
      (0, 0, 0, 0, 0, 0, fam 0 0 0, fam 0 1 0, fam 1 0 0, fam 1 1 0, ()) final) :
    Rel sm final.key (∑ i ∈ Finset.range count, lutBranchTarget (q := Q0) (p := (qd i0)) μ sp fam i)
        final.row (count * lutBranchBound393 i0 B Bh) ∧
      final.switch = fun k j ↦ fam k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q0 (R i0) 6) ↦ ∀ hk : k ≤ 3,
      Rel sm s.key (∑ i ∈ Finset.range k, lutBranchTarget (q := Q0) (p := (qd i0)) μ sp fam i) s.row
        (k * lutBranchBound393 i0 B Bh) ∧
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
    obtain ⟨hrel', hsw'⟩ := lutStep393 hsm hg' hg henc hhelp (by omega) hsw hrel hbody
    refine ⟨?_, hsw'⟩
    rw [Finset.sum_range_succ, lutBranchTarget, dif_pos (by omega), add_mul, one_mul]
    exact hrel'

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem lut393 {i0 : Nat} {model : HashModel} {key : ByteArray} {params} {o : Nat} [Fact (i0 < 8)]
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix (R i0) N 1 6} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6} {outputs : Blocks Q0 6}
    (h : Stage_online_eval.parallel_parallel_generatedRoot_290_395 Backend.backend model params i0 o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11, ()) outputs)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {z : Nat}
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {B : Nat}
    (hinput : Rel sm ![![A00, A01], ![A10, A11]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * Xe (n := N) (2 * z) * unitPair sp j) • g)
      ![c0, c1] B)
    {Bh : Nat} (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 3 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] Bh) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ (∑ i : Fin 4, scaleUpMatrix Q0 (qd i0)
        (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * z))) •
          (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i))))
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((3 * lutBranchBound393 i0 B Bh + ((6 + 6) * N * Bh * 16384 + (6 + 6) * N * B * 16384 +
        (N * 1 * ((qd i0) / 2) + (qd i0) / 2)))) := by
  obtain ⟨L, hL, hLx, -, hLd⟩ := layoutR i0 hi0
  obtain ⟨w, h⟩ := h
  unfold Stage_online_eval.parallel_parallel_generatedRoot_290_395.body at h
  extract_lets -merge _params _backend w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 at h
  simp only [Stage_online_eval.parallel_parallel_generatedRoot_290_395.constraints_0, Stage_online_eval.parallel_parallel_generatedRoot_290_395.constraints_1, Stage_online_eval.parallel_parallel_generatedRoot_290_395.constraints_2, Stage_online_eval.parallel_parallel_generatedRoot_290_395.constraints_3, Stage_online_eval.parallel_parallel_generatedRoot_290_395.constraints_4] at h
  obtain ⟨h2_0, h4_0, h6_0, h8_0, -, h18_9, h19_0, -, -, h21_0, h22_0, h21_0_1, h23_0, h24_0, h25_0, h26_0, h25_0_1, h27_0, h28_0, -, -, h29_0, -, -, h30_0, -, -, h31_0, h30_0_1, h32_0, -, -, h33_0, -, -, h34_0, -, -, h35_0, h34_0_1, h36_0, h37_0, h38_0, h39_0, h40_0, -, -, -, -, -, -, h42_0, h43_0, -, -, -, -, -, -, h44_0, h45_0, h46_0, h47_0, h48_0, -, -, -, -, -, -, h50_0, h52_0, h53_0, h54_0, -, -, -, -, -, -, h56_0, h57_0, -, -, -, -, -, -, h58_0, h59_0, h60_0, h61_0, h62_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h66_0, -, -, -, -, -, -, h68_0, -, -, -, -, -, -, h70_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h75_0, hout⟩ := h
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  rw [hwa, hwb, hwc, hwd] at h18_9
  rw [show Int.toNat 3 = 3 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := lutLoop393 (fam := fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j)
    (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]])
    (final := (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) hsm hg' hg hinput hhelp (count := 3) (by decide) h18_9
  obtain ⟨_, hw19⟩ := familyGet_eq (k := 3) h19_0
  obtain ⟨_, hw24⟩ := familyGet_eq (k := 3) h24_0
  rw [hw19] at h21_0_1
  rw [hw24] at h25_0_1
  have hA0 : w_21_0 = lutA model key Q0 6 3 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h21_0
  have hA1 : w_25_0 = lutA model key Q0 6 3 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h25_0
  have hA2 : w_22_0 = lutA model key Q0 6 3 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h22_0
  have hA3 : w_26_0 = lutA model key Q0 6 3 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h26_0
  rw [hA0] at h21_0_1
  rw [hA2] at h23_0
  rw [hA1] at h25_0_1
  rw [hA3] at h27_0
  have hhelp3 := hhelp 3
  have hsum := lut_branches (q := Q0) (q' := (R i0)) (p := (qd i0)) (m := 6) (m' := 6) (q0_eqR i0 hi0) (by inst_num)
    (by inst_num) (by inst_num) L hL hLx hsm hg' hg hinput hloop hloopSwitch hhelp3
    h29_0 h30_0 h31_0 h33_0 h34_0 h35_0 h30_0_1 h32_0 h34_0_1 h36_0 h37_0
    (Xb := ![w_38_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h52_0])
    (D' := ![w_39_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h53_0])
    (D := ![w_40_0, w_54_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h54_0])
    (M := ![![w_42_0, w_56_0], ![w_44_0, w_58_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h56_0])
    (fun j ↦ by fin_cases j; exacts [h44_0, h58_0])
    (Asw := ![![w_43_0, w_57_0], ![w_45_0, w_59_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h43_0, h57_0, h45_0, h59_0])
    (Z := ![w_46_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h60_0])
    (E' := ![w_47_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h47_0, h61_0])
    (E := ![w_48_0, w_62_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h62_0])
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (Y := ![w_70_0, w_74_0]) (X := ![w_71_0, w_75_0])
    (K := ![![w_50_0, w_64_0], ![w_66_0, w_68_0]])
    (fun j ↦ by fin_cases j; exacts [h70_0, h74_0])
    (fun j ↦ by fin_cases j; exacts [h71_0, h75_0])
    (fun j ↦ by fin_cases j; exacts [h50_0, h64_0])
    (fun j ↦ by fin_cases j; exacts [h66_0, h68_0])
  have htot := hsum
  rw [hout]
  convert htot using 1
  all_goals first
    | rfl
    | (funext a j; fin_cases a <;> fin_cases j <;> rfl)
    | (funext j; fin_cases j <;> rfl)

end RefreshTopProof.Part_I
