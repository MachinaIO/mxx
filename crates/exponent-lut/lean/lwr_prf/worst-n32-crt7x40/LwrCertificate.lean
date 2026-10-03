import LwrLut
import LwrEncrypt

/-!
# The worst-case correctness certificate

Every execution satisfying the generated claim's run relation decodes the parity of the readout of
its selected coordinates. The requirement makes the plaintext of member `k` of `a` the monomial
`X^(2 a_k)` and every block's selectors one-hot; the five blocks then accumulate `X^(2z)` for the sum
`z` of the selected coordinates, the readout LUT gives `(-1)^⌊z/16⌋ τ(z mod 16)` at the terminal
modulus, and decryption returns its parity, which is the ideal output.

## The requirement and the ideal output

The requirement accepts plaintext monomials `c X^e` of `a` with `c X^e = X^(2a)` and one-hot
selectors; the ideal output is the parity of the readout `τ(z mod 16)` of
`z = Σ_b Σ_i s_b(i) a(coords b i)`.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem ite_add_ite_eq_two (p q : Prop) [Decidable p] [Decidable q] :
    ((if p then (1 : Int) else 0) + (if q then 1 else 0) = 2) ↔ p ∧ q := by
  by_cases hp : p <;> by_cases hq : q <;> simp [hp, hq]

theorem sum_fourteen (s : Fin 14 → Int) :
    ∑ i, s i = 0 + s ⟨0, by decide⟩ + s ⟨1, by decide⟩ + s ⟨2, by decide⟩ + s ⟨3, by decide⟩ +
      s ⟨4, by decide⟩ + s ⟨5, by decide⟩ + s ⟨6, by decide⟩ + s ⟨7, by decide⟩ + s ⟨8, by decide⟩ +
      s ⟨9, by decide⟩ + s ⟨10, by decide⟩ + s ⟨11, by decide⟩ + s ⟨12, by decide⟩ +
      s ⟨13, by decide⟩ := by
  simp only [Fin.sum_univ_succ, Fin.sum_univ_zero]
  ring_nf
  rfl

set_option maxHeartbeats 64000000 in
/-- The requirement: `c_k² = 1` and `2 e_k + 32 (1 - c_k) = 4 a_k` for every member, and every block's
selectors sum to one. -/
theorem requirement_holds {ac ae a : Fin 17 → Int} {s0 s1 s2 s3 s4 : Fin 14 → Int} {out : Bool}
    (h : Requirement_0.generatedRoot GeneratedClaim.requirement_0_params
      (ac, ae, a, s0, s1, s2, s3, s4, ()) out) (ht : out = true) :
    (∀ k : Fin 17, ac k * ac k = 1 ∧ ae k * 2 + (1 - ac k) * 32 = a k * 4) ∧
      ∑ i, s0 i = 1 ∧ ∑ i, s1 i = 1 ∧ ∑ i, s2 i = 1 ∧ ∑ i, s3 i = 1 ∧ ∑ i, s4 i = 1 := by
  obtain ⟨w, h5_0, h15_0, h24_0, h33_0, h42_0, h50_0, h59_0, h68_0, h76_0, h85_0, h94_0, h102_0, h111_0, h120_0, h128_0, h137_0, h146_0, h154_0, h163_0, h172_0, h180_0, h189_0, h198_0, h206_0, h215_0, h224_0, h232_0, h241_0, h250_0, h258_0, h267_0, h276_0, h284_0, h293_0, h302_0, h310_0, h319_0, h328_0, h336_0, h345_0, h354_0, h362_0, h371_0, h380_0, h388_0, h397_0, h406_0, h414_0, h423_0, h432_0, h440_0, h451_0, h453_0, h455_0, h457_0, h459_0, h461_0, h463_0, h465_0, h467_0, h469_0, h471_0, h473_0, h475_0, h477_0, h488_0, h490_0, h492_0, h494_0, h496_0, h498_0, h500_0, h502_0, h504_0, h506_0, h508_0, h510_0, h512_0, h514_0, h525_0, h527_0, h529_0, h531_0, h533_0, h535_0, h537_0, h539_0, h541_0, h543_0, h545_0, h547_0, h549_0, h551_0, h562_0, h564_0, h566_0, h568_0, h570_0, h572_0, h574_0, h576_0, h578_0, h580_0, h582_0, h584_0, h586_0, h588_0, h599_0, h601_0, h603_0, h605_0, h607_0, h609_0, h611_0, h613_0, h615_0, h617_0, h619_0, h621_0, h623_0, h625_0, hout⟩ := h
  obtain ⟨_, e568_0⟩ := familyGetStatic_eq (k := 3) h568_0
  obtain ⟨_, e570_0⟩ := familyGetStatic_eq (k := 4) h570_0
  obtain ⟨_, e572_0⟩ := familyGetStatic_eq (k := 5) h572_0
  obtain ⟨_, e574_0⟩ := familyGetStatic_eq (k := 6) h574_0
  obtain ⟨_, e576_0⟩ := familyGetStatic_eq (k := 7) h576_0
  obtain ⟨_, e578_0⟩ := familyGetStatic_eq (k := 8) h578_0
  obtain ⟨_, e580_0⟩ := familyGetStatic_eq (k := 9) h580_0
  obtain ⟨_, e582_0⟩ := familyGetStatic_eq (k := 10) h582_0
  obtain ⟨_, e584_0⟩ := familyGetStatic_eq (k := 11) h584_0
  obtain ⟨_, e586_0⟩ := familyGetStatic_eq (k := 12) h586_0
  obtain ⟨_, e588_0⟩ := familyGetStatic_eq (k := 13) h588_0
  obtain ⟨_, e599_0⟩ := familyGetStatic_eq (k := 0) h599_0
  obtain ⟨_, e601_0⟩ := familyGetStatic_eq (k := 1) h601_0
  obtain ⟨_, e603_0⟩ := familyGetStatic_eq (k := 2) h603_0
  obtain ⟨_, e605_0⟩ := familyGetStatic_eq (k := 3) h605_0
  obtain ⟨_, e607_0⟩ := familyGetStatic_eq (k := 4) h607_0
  obtain ⟨_, e609_0⟩ := familyGetStatic_eq (k := 5) h609_0
  obtain ⟨_, e611_0⟩ := familyGetStatic_eq (k := 6) h611_0
  obtain ⟨_, e613_0⟩ := familyGetStatic_eq (k := 7) h613_0
  obtain ⟨_, e615_0⟩ := familyGetStatic_eq (k := 8) h615_0
  obtain ⟨_, e617_0⟩ := familyGetStatic_eq (k := 9) h617_0
  obtain ⟨_, e619_0⟩ := familyGetStatic_eq (k := 10) h619_0
  obtain ⟨_, e621_0⟩ := familyGetStatic_eq (k := 11) h621_0
  obtain ⟨_, e623_0⟩ := familyGetStatic_eq (k := 12) h623_0
  obtain ⟨_, e625_0⟩ := familyGetStatic_eq (k := 13) h625_0
  obtain ⟨_, e477_0⟩ := familyGetStatic_eq (k := 13) h477_0
  obtain ⟨_, e488_0⟩ := familyGetStatic_eq (k := 0) h488_0
  obtain ⟨_, e490_0⟩ := familyGetStatic_eq (k := 1) h490_0
  obtain ⟨_, e492_0⟩ := familyGetStatic_eq (k := 2) h492_0
  obtain ⟨_, e494_0⟩ := familyGetStatic_eq (k := 3) h494_0
  obtain ⟨_, e496_0⟩ := familyGetStatic_eq (k := 4) h496_0
  obtain ⟨_, e498_0⟩ := familyGetStatic_eq (k := 5) h498_0
  obtain ⟨_, e500_0⟩ := familyGetStatic_eq (k := 6) h500_0
  obtain ⟨_, e502_0⟩ := familyGetStatic_eq (k := 7) h502_0
  obtain ⟨_, e504_0⟩ := familyGetStatic_eq (k := 8) h504_0
  obtain ⟨_, e506_0⟩ := familyGetStatic_eq (k := 9) h506_0
  obtain ⟨_, e508_0⟩ := familyGetStatic_eq (k := 10) h508_0
  obtain ⟨_, e510_0⟩ := familyGetStatic_eq (k := 11) h510_0
  obtain ⟨_, e512_0⟩ := familyGetStatic_eq (k := 12) h512_0
  obtain ⟨_, e514_0⟩ := familyGetStatic_eq (k := 13) h514_0
  obtain ⟨_, e525_0⟩ := familyGetStatic_eq (k := 0) h525_0
  obtain ⟨_, e527_0⟩ := familyGetStatic_eq (k := 1) h527_0
  obtain ⟨_, e529_0⟩ := familyGetStatic_eq (k := 2) h529_0
  obtain ⟨_, e531_0⟩ := familyGetStatic_eq (k := 3) h531_0
  obtain ⟨_, e533_0⟩ := familyGetStatic_eq (k := 4) h533_0
  obtain ⟨_, e535_0⟩ := familyGetStatic_eq (k := 5) h535_0
  obtain ⟨_, e537_0⟩ := familyGetStatic_eq (k := 6) h537_0
  obtain ⟨_, e539_0⟩ := familyGetStatic_eq (k := 7) h539_0
  obtain ⟨_, e541_0⟩ := familyGetStatic_eq (k := 8) h541_0
  obtain ⟨_, e543_0⟩ := familyGetStatic_eq (k := 9) h543_0
  obtain ⟨_, e545_0⟩ := familyGetStatic_eq (k := 10) h545_0
  obtain ⟨_, e547_0⟩ := familyGetStatic_eq (k := 11) h547_0
  obtain ⟨_, e549_0⟩ := familyGetStatic_eq (k := 12) h549_0
  obtain ⟨_, e551_0⟩ := familyGetStatic_eq (k := 13) h551_0
  obtain ⟨_, e562_0⟩ := familyGetStatic_eq (k := 0) h562_0
  obtain ⟨_, e564_0⟩ := familyGetStatic_eq (k := 1) h564_0
  obtain ⟨_, e566_0⟩ := familyGetStatic_eq (k := 2) h566_0
  obtain ⟨_, e284_0⟩ := familyGetStatic_eq (k := 10) h284_0
  obtain ⟨_, e293_0⟩ := familyGetStatic_eq (k := 11) h293_0
  obtain ⟨_, e302_0⟩ := familyGetStatic_eq (k := 11) h302_0
  obtain ⟨_, e310_0⟩ := familyGetStatic_eq (k := 11) h310_0
  obtain ⟨_, e319_0⟩ := familyGetStatic_eq (k := 12) h319_0
  obtain ⟨_, e328_0⟩ := familyGetStatic_eq (k := 12) h328_0
  obtain ⟨_, e336_0⟩ := familyGetStatic_eq (k := 12) h336_0
  obtain ⟨_, e345_0⟩ := familyGetStatic_eq (k := 13) h345_0
  obtain ⟨_, e354_0⟩ := familyGetStatic_eq (k := 13) h354_0
  obtain ⟨_, e362_0⟩ := familyGetStatic_eq (k := 13) h362_0
  obtain ⟨_, e371_0⟩ := familyGetStatic_eq (k := 14) h371_0
  obtain ⟨_, e380_0⟩ := familyGetStatic_eq (k := 14) h380_0
  obtain ⟨_, e388_0⟩ := familyGetStatic_eq (k := 14) h388_0
  obtain ⟨_, e397_0⟩ := familyGetStatic_eq (k := 15) h397_0
  obtain ⟨_, e406_0⟩ := familyGetStatic_eq (k := 15) h406_0
  obtain ⟨_, e414_0⟩ := familyGetStatic_eq (k := 15) h414_0
  obtain ⟨_, e423_0⟩ := familyGetStatic_eq (k := 16) h423_0
  obtain ⟨_, e432_0⟩ := familyGetStatic_eq (k := 16) h432_0
  obtain ⟨_, e440_0⟩ := familyGetStatic_eq (k := 16) h440_0
  obtain ⟨_, e451_0⟩ := familyGetStatic_eq (k := 0) h451_0
  obtain ⟨_, e453_0⟩ := familyGetStatic_eq (k := 1) h453_0
  obtain ⟨_, e455_0⟩ := familyGetStatic_eq (k := 2) h455_0
  obtain ⟨_, e457_0⟩ := familyGetStatic_eq (k := 3) h457_0
  obtain ⟨_, e459_0⟩ := familyGetStatic_eq (k := 4) h459_0
  obtain ⟨_, e461_0⟩ := familyGetStatic_eq (k := 5) h461_0
  obtain ⟨_, e463_0⟩ := familyGetStatic_eq (k := 6) h463_0
  obtain ⟨_, e465_0⟩ := familyGetStatic_eq (k := 7) h465_0
  obtain ⟨_, e467_0⟩ := familyGetStatic_eq (k := 8) h467_0
  obtain ⟨_, e469_0⟩ := familyGetStatic_eq (k := 9) h469_0
  obtain ⟨_, e471_0⟩ := familyGetStatic_eq (k := 10) h471_0
  obtain ⟨_, e473_0⟩ := familyGetStatic_eq (k := 11) h473_0
  obtain ⟨_, e475_0⟩ := familyGetStatic_eq (k := 12) h475_0
  obtain ⟨_, e5_0⟩ := familyGetStatic_eq (k := 0) h5_0
  obtain ⟨_, e15_0⟩ := familyGetStatic_eq (k := 0) h15_0
  obtain ⟨_, e24_0⟩ := familyGetStatic_eq (k := 0) h24_0
  obtain ⟨_, e33_0⟩ := familyGetStatic_eq (k := 1) h33_0
  obtain ⟨_, e42_0⟩ := familyGetStatic_eq (k := 1) h42_0
  obtain ⟨_, e50_0⟩ := familyGetStatic_eq (k := 1) h50_0
  obtain ⟨_, e59_0⟩ := familyGetStatic_eq (k := 2) h59_0
  obtain ⟨_, e68_0⟩ := familyGetStatic_eq (k := 2) h68_0
  obtain ⟨_, e76_0⟩ := familyGetStatic_eq (k := 2) h76_0
  obtain ⟨_, e85_0⟩ := familyGetStatic_eq (k := 3) h85_0
  obtain ⟨_, e94_0⟩ := familyGetStatic_eq (k := 3) h94_0
  obtain ⟨_, e102_0⟩ := familyGetStatic_eq (k := 3) h102_0
  obtain ⟨_, e111_0⟩ := familyGetStatic_eq (k := 4) h111_0
  obtain ⟨_, e120_0⟩ := familyGetStatic_eq (k := 4) h120_0
  obtain ⟨_, e128_0⟩ := familyGetStatic_eq (k := 4) h128_0
  obtain ⟨_, e137_0⟩ := familyGetStatic_eq (k := 5) h137_0
  obtain ⟨_, e146_0⟩ := familyGetStatic_eq (k := 5) h146_0
  obtain ⟨_, e154_0⟩ := familyGetStatic_eq (k := 5) h154_0
  obtain ⟨_, e163_0⟩ := familyGetStatic_eq (k := 6) h163_0
  obtain ⟨_, e172_0⟩ := familyGetStatic_eq (k := 6) h172_0
  obtain ⟨_, e180_0⟩ := familyGetStatic_eq (k := 6) h180_0
  obtain ⟨_, e189_0⟩ := familyGetStatic_eq (k := 7) h189_0
  obtain ⟨_, e198_0⟩ := familyGetStatic_eq (k := 7) h198_0
  obtain ⟨_, e206_0⟩ := familyGetStatic_eq (k := 7) h206_0
  obtain ⟨_, e215_0⟩ := familyGetStatic_eq (k := 8) h215_0
  obtain ⟨_, e224_0⟩ := familyGetStatic_eq (k := 8) h224_0
  obtain ⟨_, e232_0⟩ := familyGetStatic_eq (k := 8) h232_0
  obtain ⟨_, e241_0⟩ := familyGetStatic_eq (k := 9) h241_0
  obtain ⟨_, e250_0⟩ := familyGetStatic_eq (k := 9) h250_0
  obtain ⟨_, e258_0⟩ := familyGetStatic_eq (k := 9) h258_0
  obtain ⟨_, e267_0⟩ := familyGetStatic_eq (k := 10) h267_0
  obtain ⟨_, e276_0⟩ := familyGetStatic_eq (k := 10) h276_0
  subst ht
  simp only [e568_0, e570_0, e572_0, e574_0, e576_0, e578_0, e580_0, e582_0, e584_0, e586_0, e588_0, e599_0, e601_0, e603_0, e605_0, e607_0, e609_0, e611_0, e613_0, e615_0, e617_0, e619_0, e621_0, e623_0, e625_0, e477_0, e488_0, e490_0, e492_0, e494_0, e496_0, e498_0, e500_0, e502_0, e504_0, e506_0, e508_0, e510_0, e512_0, e514_0, e525_0, e527_0, e529_0, e531_0, e533_0, e535_0, e537_0, e539_0, e541_0, e543_0, e545_0, e547_0, e549_0, e551_0, e562_0, e564_0, e566_0, e284_0, e293_0, e302_0, e310_0, e319_0, e328_0, e336_0, e345_0, e354_0, e362_0, e371_0, e380_0, e388_0, e397_0, e406_0, e414_0, e423_0, e432_0, e440_0, e451_0, e453_0, e455_0, e457_0, e459_0, e461_0, e463_0, e465_0, e467_0, e469_0, e471_0, e473_0, e475_0, e5_0, e15_0, e24_0, e33_0, e42_0, e50_0, e59_0, e68_0, e76_0, e85_0, e94_0, e102_0, e111_0, e120_0, e128_0, e137_0, e146_0, e154_0, e163_0, e172_0, e180_0, e189_0, e198_0, e206_0, e215_0, e224_0, e232_0, e241_0, e250_0, e258_0, e267_0, e276_0] at hout
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨-, m0a⟩, m0b⟩, m1a⟩, m1b⟩, m2a⟩, m2b⟩, m3a⟩, m3b⟩, m4a⟩, m4b⟩, m5a⟩, m5b⟩, m6a⟩, m6b⟩, m7a⟩, m7b⟩, m8a⟩, m8b⟩, m9a⟩, m9b⟩, m10a⟩, m10b⟩, m11a⟩, m11b⟩, m12a⟩, m12b⟩, m13a⟩, m13b⟩, m14a⟩, m14b⟩, m15a⟩, m15b⟩, m16a⟩, m16b⟩, sum0⟩, sum1⟩, sum2⟩, sum3⟩, sum4⟩ := hout
  refine ⟨fun k ↦ ?_, by rw [sum_fourteen]; exact sum0, by rw [sum_fourteen]; exact sum1,
    by rw [sum_fourteen]; exact sum2, by rw [sum_fourteen]; exact sum3,
    by rw [sum_fourteen]; exact sum4⟩
  fin_cases k
  exacts [⟨m0a, m0b⟩, ⟨m1a, m1b⟩, ⟨m2a, m2b⟩, ⟨m3a, m3b⟩, ⟨m4a, m4b⟩, ⟨m5a, m5b⟩, ⟨m6a, m6b⟩, ⟨m7a, m7b⟩, ⟨m8a, m8b⟩, ⟨m9a, m9b⟩, ⟨m10a, m10b⟩, ⟨m11a, m11b⟩, ⟨m12a, m12b⟩, ⟨m13a, m13b⟩, ⟨m14a, m14b⟩, ⟨m15a, m15b⟩, ⟨m16a, m16b⟩]

/-- The sum `z = Σ_b Σ_i s_b(i) a(coords b i)` of the selected coordinates. -/
def selectedSum (a : Fin 17 → Int) (s : Fin 5 → Fin 14 → Int) : Int :=
  ∑ b : Fin 5, ∑ i : Fin 14, s b i * a ⟨coords b i, coords_lt b i⟩

set_option maxHeartbeats 64000000 in
/-- The ideal output: the parity of `τ(z mod 16) = ⌊2 (z mod 16) / 16⌋ + 1`. -/
theorem ideal_value {a : Fin 17 → Int} {s0 s1 s2 s3 s4 : Fin 14 → Int} {out : Int}
    (h : Ideal.generatedRoot GeneratedClaim.ideal_params (s0, a, s1, s2, s3, s4, ()) out) :
    out = (selectedSum a ![s0, s1, s2, s3, s4] % 16 * 2 / 16 + 1) % 2 := by
  obtain ⟨w, h2_0, h4_0, h7_0, h8_0, h11_0, h12_0, h15_0, h16_0, h19_0, h20_0, h23_0, h24_0, h27_0, h28_0, h31_0, h32_0, h35_0, h36_0, h39_0, h40_0, h43_0, h44_0, h47_0, h48_0, h51_0, h52_0, h55_0, h56_0, h60_0, h61_0, h64_0, h65_0, h68_0, h69_0, h72_0, h73_0, h76_0, h77_0, h80_0, h81_0, h84_0, h85_0, h88_0, h89_0, h92_0, h93_0, h96_0, h97_0, h100_0, h101_0, h104_0, h105_0, h108_0, h109_0, h112_0, h113_0, h117_0, h118_0, h121_0, h122_0, h125_0, h126_0, h129_0, h130_0, h133_0, h134_0, h137_0, h138_0, h141_0, h142_0, h145_0, h146_0, h149_0, h150_0, h153_0, h154_0, h157_0, h158_0, h161_0, h162_0, h165_0, h166_0, h169_0, h170_0, h174_0, h175_0, h178_0, h179_0, h182_0, h183_0, h186_0, h187_0, h190_0, h191_0, h194_0, h195_0, h198_0, h199_0, h202_0, h203_0, h206_0, h207_0, h210_0, h211_0, h214_0, h215_0, h218_0, h219_0, h222_0, h223_0, h226_0, h227_0, h231_0, h232_0, h235_0, h236_0, h239_0, h240_0, h243_0, h244_0, h247_0, h248_0, h251_0, h252_0, h255_0, h256_0, h259_0, h260_0, h263_0, h264_0, h267_0, h268_0, h271_0, h272_0, h275_0, h276_0, h279_0, h280_0, h283_0, h284_0, h287_0, h291_0, h295_0, hout⟩ := h
  obtain ⟨_, e263_0⟩ := familyGetStatic_eq (k := 8) h263_0
  obtain ⟨_, e264_0⟩ := familyGetStatic_eq (k := 16) h264_0
  obtain ⟨_, e267_0⟩ := familyGetStatic_eq (k := 9) h267_0
  obtain ⟨_, e268_0⟩ := familyGetStatic_eq (k := 16) h268_0
  obtain ⟨_, e271_0⟩ := familyGetStatic_eq (k := 10) h271_0
  obtain ⟨_, e272_0⟩ := familyGetStatic_eq (k := 16) h272_0
  obtain ⟨_, e275_0⟩ := familyGetStatic_eq (k := 11) h275_0
  obtain ⟨_, e276_0⟩ := familyGetStatic_eq (k := 16) h276_0
  obtain ⟨_, e279_0⟩ := familyGetStatic_eq (k := 12) h279_0
  obtain ⟨_, e280_0⟩ := familyGetStatic_eq (k := 16) h280_0
  obtain ⟨_, e283_0⟩ := familyGetStatic_eq (k := 13) h283_0
  obtain ⟨_, e284_0⟩ := familyGetStatic_eq (k := 16) h284_0
  obtain ⟨_, e198_0⟩ := familyGetStatic_eq (k := 6) h198_0
  obtain ⟨_, e199_0⟩ := familyGetStatic_eq (k := 6) h199_0
  obtain ⟨_, e202_0⟩ := familyGetStatic_eq (k := 7) h202_0
  obtain ⟨_, e203_0⟩ := familyGetStatic_eq (k := 7) h203_0
  obtain ⟨_, e206_0⟩ := familyGetStatic_eq (k := 8) h206_0
  obtain ⟨_, e207_0⟩ := familyGetStatic_eq (k := 8) h207_0
  obtain ⟨_, e210_0⟩ := familyGetStatic_eq (k := 9) h210_0
  obtain ⟨_, e211_0⟩ := familyGetStatic_eq (k := 9) h211_0
  obtain ⟨_, e214_0⟩ := familyGetStatic_eq (k := 10) h214_0
  obtain ⟨_, e215_0⟩ := familyGetStatic_eq (k := 10) h215_0
  obtain ⟨_, e218_0⟩ := familyGetStatic_eq (k := 11) h218_0
  obtain ⟨_, e219_0⟩ := familyGetStatic_eq (k := 11) h219_0
  obtain ⟨_, e222_0⟩ := familyGetStatic_eq (k := 12) h222_0
  obtain ⟨_, e223_0⟩ := familyGetStatic_eq (k := 13) h223_0
  obtain ⟨_, e226_0⟩ := familyGetStatic_eq (k := 13) h226_0
  obtain ⟨_, e227_0⟩ := familyGetStatic_eq (k := 16) h227_0
  obtain ⟨_, e231_0⟩ := familyGetStatic_eq (k := 0) h231_0
  obtain ⟨_, e232_0⟩ := familyGetStatic_eq (k := 3) h232_0
  obtain ⟨_, e235_0⟩ := familyGetStatic_eq (k := 1) h235_0
  obtain ⟨_, e236_0⟩ := familyGetStatic_eq (k := 6) h236_0
  obtain ⟨_, e239_0⟩ := familyGetStatic_eq (k := 2) h239_0
  obtain ⟨_, e240_0⟩ := familyGetStatic_eq (k := 7) h240_0
  obtain ⟨_, e243_0⟩ := familyGetStatic_eq (k := 3) h243_0
  obtain ⟨_, e244_0⟩ := familyGetStatic_eq (k := 13) h244_0
  obtain ⟨_, e247_0⟩ := familyGetStatic_eq (k := 4) h247_0
  obtain ⟨_, e248_0⟩ := familyGetStatic_eq (k := 14) h248_0
  obtain ⟨_, e251_0⟩ := familyGetStatic_eq (k := 5) h251_0
  obtain ⟨_, e252_0⟩ := familyGetStatic_eq (k := 16) h252_0
  obtain ⟨_, e255_0⟩ := familyGetStatic_eq (k := 6) h255_0
  obtain ⟨_, e256_0⟩ := familyGetStatic_eq (k := 16) h256_0
  obtain ⟨_, e259_0⟩ := familyGetStatic_eq (k := 7) h259_0
  obtain ⟨_, e260_0⟩ := familyGetStatic_eq (k := 16) h260_0
  obtain ⟨_, e133_0⟩ := familyGetStatic_eq (k := 4) h133_0
  obtain ⟨_, e134_0⟩ := familyGetStatic_eq (k := 5) h134_0
  obtain ⟨_, e137_0⟩ := familyGetStatic_eq (k := 5) h137_0
  obtain ⟨_, e138_0⟩ := familyGetStatic_eq (k := 8) h138_0
  obtain ⟨_, e141_0⟩ := familyGetStatic_eq (k := 6) h141_0
  obtain ⟨_, e142_0⟩ := familyGetStatic_eq (k := 9) h142_0
  obtain ⟨_, e145_0⟩ := familyGetStatic_eq (k := 7) h145_0
  obtain ⟨_, e146_0⟩ := familyGetStatic_eq (k := 10) h146_0
  obtain ⟨_, e149_0⟩ := familyGetStatic_eq (k := 8) h149_0
  obtain ⟨_, e150_0⟩ := familyGetStatic_eq (k := 11) h150_0
  obtain ⟨_, e153_0⟩ := familyGetStatic_eq (k := 9) h153_0
  obtain ⟨_, e154_0⟩ := familyGetStatic_eq (k := 12) h154_0
  obtain ⟨_, e157_0⟩ := familyGetStatic_eq (k := 10) h157_0
  obtain ⟨_, e158_0⟩ := familyGetStatic_eq (k := 15) h158_0
  obtain ⟨_, e161_0⟩ := familyGetStatic_eq (k := 11) h161_0
  obtain ⟨_, e162_0⟩ := familyGetStatic_eq (k := 16) h162_0
  obtain ⟨_, e165_0⟩ := familyGetStatic_eq (k := 12) h165_0
  obtain ⟨_, e166_0⟩ := familyGetStatic_eq (k := 16) h166_0
  obtain ⟨_, e169_0⟩ := familyGetStatic_eq (k := 13) h169_0
  obtain ⟨_, e170_0⟩ := familyGetStatic_eq (k := 16) h170_0
  obtain ⟨_, e174_0⟩ := familyGetStatic_eq (k := 0) h174_0
  obtain ⟨_, e175_0⟩ := familyGetStatic_eq (k := 0) h175_0
  obtain ⟨_, e178_0⟩ := familyGetStatic_eq (k := 1) h178_0
  obtain ⟨_, e179_0⟩ := familyGetStatic_eq (k := 1) h179_0
  obtain ⟨_, e182_0⟩ := familyGetStatic_eq (k := 2) h182_0
  obtain ⟨_, e183_0⟩ := familyGetStatic_eq (k := 2) h183_0
  obtain ⟨_, e186_0⟩ := familyGetStatic_eq (k := 3) h186_0
  obtain ⟨_, e187_0⟩ := familyGetStatic_eq (k := 3) h187_0
  obtain ⟨_, e190_0⟩ := familyGetStatic_eq (k := 4) h190_0
  obtain ⟨_, e191_0⟩ := familyGetStatic_eq (k := 4) h191_0
  obtain ⟨_, e194_0⟩ := familyGetStatic_eq (k := 5) h194_0
  obtain ⟨_, e195_0⟩ := familyGetStatic_eq (k := 5) h195_0
  obtain ⟨_, e68_0⟩ := familyGetStatic_eq (k := 2) h68_0
  obtain ⟨_, e69_0⟩ := familyGetStatic_eq (k := 2) h69_0
  obtain ⟨_, e72_0⟩ := familyGetStatic_eq (k := 3) h72_0
  obtain ⟨_, e73_0⟩ := familyGetStatic_eq (k := 4) h73_0
  obtain ⟨_, e76_0⟩ := familyGetStatic_eq (k := 4) h76_0
  obtain ⟨_, e77_0⟩ := familyGetStatic_eq (k := 6) h77_0
  obtain ⟨_, e80_0⟩ := familyGetStatic_eq (k := 5) h80_0
  obtain ⟨_, e81_0⟩ := familyGetStatic_eq (k := 7) h81_0
  obtain ⟨_, e84_0⟩ := familyGetStatic_eq (k := 6) h84_0
  obtain ⟨_, e85_0⟩ := familyGetStatic_eq (k := 8) h85_0
  obtain ⟨_, e88_0⟩ := familyGetStatic_eq (k := 7) h88_0
  obtain ⟨_, e89_0⟩ := familyGetStatic_eq (k := 9) h89_0
  obtain ⟨_, e92_0⟩ := familyGetStatic_eq (k := 8) h92_0
  obtain ⟨_, e93_0⟩ := familyGetStatic_eq (k := 12) h93_0
  obtain ⟨_, e96_0⟩ := familyGetStatic_eq (k := 9) h96_0
  obtain ⟨_, e97_0⟩ := familyGetStatic_eq (k := 13) h97_0
  obtain ⟨_, e100_0⟩ := familyGetStatic_eq (k := 10) h100_0
  obtain ⟨_, e101_0⟩ := familyGetStatic_eq (k := 14) h101_0
  obtain ⟨_, e104_0⟩ := familyGetStatic_eq (k := 11) h104_0
  obtain ⟨_, e105_0⟩ := familyGetStatic_eq (k := 15) h105_0
  obtain ⟨_, e108_0⟩ := familyGetStatic_eq (k := 12) h108_0
  obtain ⟨_, e109_0⟩ := familyGetStatic_eq (k := 16) h109_0
  obtain ⟨_, e112_0⟩ := familyGetStatic_eq (k := 13) h112_0
  obtain ⟨_, e113_0⟩ := familyGetStatic_eq (k := 16) h113_0
  obtain ⟨_, e117_0⟩ := familyGetStatic_eq (k := 0) h117_0
  obtain ⟨_, e118_0⟩ := familyGetStatic_eq (k := 0) h118_0
  obtain ⟨_, e121_0⟩ := familyGetStatic_eq (k := 1) h121_0
  obtain ⟨_, e122_0⟩ := familyGetStatic_eq (k := 1) h122_0
  obtain ⟨_, e125_0⟩ := familyGetStatic_eq (k := 2) h125_0
  obtain ⟨_, e126_0⟩ := familyGetStatic_eq (k := 2) h126_0
  obtain ⟨_, e129_0⟩ := familyGetStatic_eq (k := 3) h129_0
  obtain ⟨_, e130_0⟩ := familyGetStatic_eq (k := 3) h130_0
  obtain ⟨_, e2_0⟩ := familyGetStatic_eq (k := 0) h2_0
  obtain ⟨_, e4_0⟩ := familyGetStatic_eq (k := 4) h4_0
  obtain ⟨_, e7_0⟩ := familyGetStatic_eq (k := 1) h7_0
  obtain ⟨_, e8_0⟩ := familyGetStatic_eq (k := 5) h8_0
  obtain ⟨_, e11_0⟩ := familyGetStatic_eq (k := 2) h11_0
  obtain ⟨_, e12_0⟩ := familyGetStatic_eq (k := 10) h12_0
  obtain ⟨_, e15_0⟩ := familyGetStatic_eq (k := 3) h15_0
  obtain ⟨_, e16_0⟩ := familyGetStatic_eq (k := 11) h16_0
  obtain ⟨_, e19_0⟩ := familyGetStatic_eq (k := 4) h19_0
  obtain ⟨_, e20_0⟩ := familyGetStatic_eq (k := 12) h20_0
  obtain ⟨_, e23_0⟩ := familyGetStatic_eq (k := 5) h23_0
  obtain ⟨_, e24_0⟩ := familyGetStatic_eq (k := 14) h24_0
  obtain ⟨_, e27_0⟩ := familyGetStatic_eq (k := 6) h27_0
  obtain ⟨_, e28_0⟩ := familyGetStatic_eq (k := 15) h28_0
  obtain ⟨_, e31_0⟩ := familyGetStatic_eq (k := 7) h31_0
  obtain ⟨_, e32_0⟩ := familyGetStatic_eq (k := 16) h32_0
  obtain ⟨_, e35_0⟩ := familyGetStatic_eq (k := 8) h35_0
  obtain ⟨_, e36_0⟩ := familyGetStatic_eq (k := 16) h36_0
  obtain ⟨_, e39_0⟩ := familyGetStatic_eq (k := 9) h39_0
  obtain ⟨_, e40_0⟩ := familyGetStatic_eq (k := 16) h40_0
  obtain ⟨_, e43_0⟩ := familyGetStatic_eq (k := 10) h43_0
  obtain ⟨_, e44_0⟩ := familyGetStatic_eq (k := 16) h44_0
  obtain ⟨_, e47_0⟩ := familyGetStatic_eq (k := 11) h47_0
  obtain ⟨_, e48_0⟩ := familyGetStatic_eq (k := 16) h48_0
  obtain ⟨_, e51_0⟩ := familyGetStatic_eq (k := 12) h51_0
  obtain ⟨_, e52_0⟩ := familyGetStatic_eq (k := 16) h52_0
  obtain ⟨_, e55_0⟩ := familyGetStatic_eq (k := 13) h55_0
  obtain ⟨_, e56_0⟩ := familyGetStatic_eq (k := 16) h56_0
  obtain ⟨_, e60_0⟩ := familyGetStatic_eq (k := 0) h60_0
  obtain ⟨_, e61_0⟩ := familyGetStatic_eq (k := 0) h61_0
  obtain ⟨_, e64_0⟩ := familyGetStatic_eq (k := 1) h64_0
  obtain ⟨_, e65_0⟩ := familyGetStatic_eq (k := 1) h65_0
  simp only [e263_0, e264_0, e267_0, e268_0, e271_0, e272_0, e275_0, e276_0, e279_0, e280_0, e283_0, e284_0, e198_0, e199_0, e202_0, e203_0, e206_0, e207_0, e210_0, e211_0, e214_0, e215_0, e218_0, e219_0, e222_0, e223_0, e226_0, e227_0, e231_0, e232_0, e235_0, e236_0, e239_0, e240_0, e243_0, e244_0, e247_0, e248_0, e251_0, e252_0, e255_0, e256_0, e259_0, e260_0, e133_0, e134_0, e137_0, e138_0, e141_0, e142_0, e145_0, e146_0, e149_0, e150_0, e153_0, e154_0, e157_0, e158_0, e161_0, e162_0, e165_0, e166_0, e169_0, e170_0, e174_0, e175_0, e178_0, e179_0, e182_0, e183_0, e186_0, e187_0, e190_0, e191_0, e194_0, e195_0, e68_0, e69_0, e72_0, e73_0, e76_0, e77_0, e80_0, e81_0, e84_0, e85_0, e88_0, e89_0, e92_0, e93_0, e96_0, e97_0, e100_0, e101_0, e104_0, e105_0, e108_0, e109_0, e112_0, e113_0, e117_0, e118_0, e121_0, e122_0, e125_0, e126_0, e129_0, e130_0, e2_0, e4_0, e7_0, e8_0, e11_0, e12_0, e15_0, e16_0, e19_0, e20_0, e23_0, e24_0, e27_0, e28_0, e31_0, e32_0, e35_0, e36_0, e39_0, e40_0, e43_0, e44_0, e47_0, e48_0, e51_0, e52_0, e55_0, e56_0, e60_0, e61_0, e64_0, e65_0] at hout
  rw [hout]
  congr 3
  simp only [selectedSum, Fin.sum_univ_succ, Fin.sum_univ_zero, coords]
  simp
  ring_nf

/-! ## Plaintexts of the run -/

/-- A one-hot family selects one cell. -/
theorem one_hot {s : Fin 14 → Int} (h01 : ∀ i, 0 ≤ s i ∧ s i ≤ 1) (hsum : ∑ i, s i = 1) :
    ∃ sel : Fin 14, ∀ i, s i = if i = sel then 1 else 0 := by
  have hex : ∃ sel, s sel = 1 := by
    by_contra hne
    push_neg at hne
    have hzero : ∀ i, s i = 0 := fun i ↦ by have := h01 i; have := hne i; omega
    simp [hzero] at hsum
  obtain ⟨sel, hsel⟩ := hex
  refine ⟨sel, fun i ↦ ?_⟩
  split
  · rename_i hi
    rw [hi, hsel]
  · rename_i hi
    by_contra hne
    have hi1 : s i = 1 := by have := h01 i; omega
    have hpair : ∑ j ∈ ({i, sel} : Finset (Fin 14)), s j ≤ ∑ j, s j :=
      Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
        (fun j _ _ ↦ (h01 j).1)
    rw [Finset.sum_pair hi, hi1, hsel, hsum] at hpair
    omega

/-- The block sum with one-hot selectors is the selected cell. -/
theorem collapse {q m p : Nat} (h : ExactMatrix q N 1 m) (sp μ : ErrorPoly N)
    (α ν : Fin 14 → ErrorPoly N) (s : Fin 14 → Int) (sel : Fin 14)
    (hs : ∀ i, s i = if i = sel then 1 else 0) (hν : ∀ i, ν i = (s i : ErrorPoly N)) :
    (∑ i : Fin 14, fun j ↦ reducePoly q N (((p : Nat) : Int) * (α i * μ * ν i) * unitPair sp j) • h) =
      fun j ↦ reducePoly q N (((p : Nat) : Int) * (α sel * μ) * unitPair sp j) • h := by
  funext j
  rw [Finset.sum_apply, Finset.sum_eq_single sel]
  · simp [hν, hs]
  · intro i _ hi
    simp [hν, hs, hi]
  · simp

/-- A one-hot block selects one coordinate. -/
theorem one_hot_sum (a : Fin 17 → Int) (b : Fin 5) {s : Fin 14 → Int} {sel : Fin 14}
    (hs : ∀ i, s i = if i = sel then 1 else 0) :
    ∑ i : Fin 14, s i * a ⟨coords b i, coords_lt b i⟩ = a ⟨coords b sel, coords_lt b sel⟩ := by
  rw [Finset.sum_eq_single sel]
  · simp [hs]
  · intro i _ hi
    simp [hs, hi]
  · simp

/-- The plaintext of a member of `a` accepted by the requirement is `X^(2a)`. -/
theorem message_member {c e a : Int} (hc : c * c = 1) (hrel : e * 2 + (1 - c) * 32 = a * 4)
    (he : 0 ≤ e ∧ e ≤ 31) : message (n := N) c e = Xe (2 * a.toNat) := by
  have hc' : c = 1 ∨ c = -1 := Int.eq_one_or_neg_one_of_mul_eq_one hc
  rcases hc' with rfl | rfl
  · exact message_one rfl (by omega) (by simp only [N]; omega)
  · rw [message, show ((2 * N : Nat) : Int) = 64 from rfl, Int.emod_eq_of_lt (by omega) (by omega)]
    have hsplit : 2 * a.toNat = e.toNat + 32 := by omega
    rw [Xe, hsplit, pow_add, Negacyclic.root_pow_n]
    push_cast
    ring

/-- A selector plaintext with exponent zero is the bit. -/
theorem message_selector {s x : Int} (hx : x = 0) : message (n := N) s x = (s : ErrorPoly N) := by
  subst hx
  simp [message]

theorem Xe_mul (a b : Nat) : Xe (n := N) a * Xe b = Xe (a + b) := by
  rw [Xe, Xe, Xe, pow_add]

theorem intCast_coeff_zero (x : Int) : ((x : ErrorPoly N)).coeff ⟨0, hN⟩ = x := by
  have h := intCast_mul_coeff x (1 : ErrorPoly N) ⟨0, hN⟩
  rw [mul_one] at h
  have h1 := Negacyclic.coeff_root_pow (R := Int) (n := N) hN ⟨0, hN⟩ ⟨0, hN⟩
  rw [show ((⟨0, hN⟩ : Fin N) : Nat) = 0 from rfl, pow_zero, if_pos rfl] at h1
  rw [h, h1, mul_one]

/-- The decrypted parity of the readout is the parity of `τ(z mod 16)`. -/
theorem readout_parity (z : Nat) :
    ((lutValue z).coeff ⟨0, hN⟩) % 2 = (((z : Int) % 16) * 2 / 16 + 1) % 2 := by
  rw [lutValue, intCast_coeff_zero, MxxExponentLutLwr.readout]
  have hτ : ((2 * (z % 16) / 16 + 1 : Nat) : Int) = ((z : Int) % 16) * 2 / 16 + 1 := by omega
  rw [hτ]
  rcases Nat.even_or_odd (z / 16) with h | h
  · rw [h.neg_one_pow, one_mul]
  · rw [h.neg_one_pow, neg_one_mul, Int.neg_emod_two]

theorem lutValue_coeff_bound (z : Nat) :
    -2 ≤ (lutValue z).coeff ⟨0, hN⟩ ∧ (lutValue z).coeff ⟨0, hN⟩ ≤ 2 := by
  rw [lutValue, intCast_coeff_zero, MxxExponentLutLwr.readout]
  have hτ : 2 * (z % 16) / 16 + 1 ≤ 2 := by omega
  rcases Nat.even_or_odd (z / 16) with h | h
  · rw [h.neg_one_pow]; omega
  · rw [h.neg_one_pow]; omega

theorem secret_ternary {out : ExactMatrix Q7 N 1 1}
    (h : Stage_payload.generatedRoot GeneratedClaim.stage_0_params () out) :
    ∃ s : ErrorPoly N, out = (fun _ _ ↦ reducePoly Q7 N s) ∧ polyNorm s ≤ 1 := by
  obtain ⟨w, hw, hout⟩ := h
  obtain ⟨s, hs, hn⟩ := ternary_poly hw
  exact ⟨s, hout.trans hs, hn⟩

theorem mask_ternary {out : ExactMatrix Q7 N 1 1}
    (h : Stage_mask.generatedRoot GeneratedClaim.stage_1_params () out) :
    ∃ s : ErrorPoly N, out = (fun _ _ ↦ reducePoly Q7 N s) ∧ polyNorm s ≤ 1 := by
  obtain ⟨w, hw, hout⟩ := h
  obtain ⟨s, hs, hn⟩ := ternary_poly hw
  exact ⟨s, hout.trans hs, hn⟩

set_option maxHeartbeats 64000000 in
/-- Every run decodes the parity of the readout of its selected coordinates. -/
theorem correct {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (hruns : GeneratedClaim.Runs model external execution) :
    execution.«stage_5» = execution.«ideal» := by
  obtain ⟨hvalid, hpay, hmask, hpre, henc, honline, hdec, hreq, hideal, hreqtrue⟩ := hruns
  obtain ⟨sp, hsp, hspn⟩ := secret_ternary hpay
  obtain ⟨sm, hsm, hsmn⟩ := mask_ternary hmask
  obtain ⟨-, ⟨hc1, hc2⟩, ⟨he1, he2⟩, ha, hac, hae, hs0, hx0, hs1, hx1, hs2, hx2, hs3, hx3, hs4, hx4⟩ :=
    hvalid
  obtain ⟨hmem, hsum0, hsum1, hsum2, hsum3, hsum4⟩ := requirement_holds hreq hreqtrue
  have hid := ideal_value hideal
  set a := external.input_3
  set s0 := external.input_6
  set s1 := external.input_8
  set s2 := external.input_10
  set s3 := external.input_12
  set s4 := external.input_14
  set x0 := external.input_7
  set x1 := external.input_9
  set x2 := external.input_11
  set x3 := external.input_13
  set x4 := external.input_15
  -- The selected cells and the plaintexts.
  obtain ⟨sel0, hsel0⟩ := one_hot hs0 hsum0
  obtain ⟨sel1, hsel1⟩ := one_hot hs1 hsum1
  obtain ⟨sel2, hsel2⟩ := one_hot hs2 hsum2
  obtain ⟨sel3, hsel3⟩ := one_hot hs3 hsum3
  obtain ⟨sel4, hsel4⟩ := one_hot hs4 hsum4
  let α : Fin 17 → ErrorPoly N := fun k ↦ Xe (2 * (a k).toNat)
  have hα : ∀ k, message (n := N) (external.input_4 k) (external.input_5 k) = α k := fun k ↦
    message_member (hmem k).1 (hmem k).2 (hae k)
  have hν0 : ∀ i, message (n := N) (s0 i) (x0 i) = (s0 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx0 i; omega)
  have hν1 : ∀ i, message (n := N) (s1 i) (x1 i) = (s1 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx1 i; omega)
  have hν2 : ∀ i, message (n := N) (s2 i) (x2 i) = (s2 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx2 i; omega)
  have hν3 : ∀ i, message (n := N) (s3 i) (x3 i) = (s3 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx3 i; omega)
  have hν4 : ∀ i, message (n := N) (s4 i) (x4 i) = (s4 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx4 i; omega)
  have hμ0 : message (n := N) external.input_1 external.input_2 = Xe 0 :=
    message_one (k := 0) (by omega) (by simp; omega) (by simp only [N]; omega)
  -- The gadget matrices.
  obtain ⟨G14, hG14⟩ := gadget_exists (m := 14) layout7 rfl rfl rfl
  obtain ⟨g12, hg12⟩ := gadget_exists (m := 12) layout6 rfl rfl rfl
  obtain ⟨g10, hg10⟩ := gadget_exists (m := 10) layout5 rfl rfl rfl
  obtain ⟨g8, hg8⟩ := gadget_exists (m := 8) layout4 rfl rfl rfl
  obtain ⟨g6, hg6⟩ := gadget_exists (m := 6) layout3 rfl rfl rfl
  obtain ⟨g4, hg4⟩ := gadget_exists (m := 4) layout2 rfl rfl rfl
  obtain ⟨g2, hg2⟩ := gadget_exists (m := 2) layout1 rfl rfl rfl
  have hg12u : centeredRebaseRuns g12 (centeredRebase g12 : ExactMatrix Q7 N 1 12) := ⟨hQ7, rfl⟩
  have hg10u : centeredRebaseRuns g10 (centeredRebase g10 : ExactMatrix Q6 N 1 10) := ⟨hQ6, rfl⟩
  have hg8u : centeredRebaseRuns g8 (centeredRebase g8 : ExactMatrix Q5 N 1 8) := ⟨hQ5, rfl⟩
  have hg6u : centeredRebaseRuns g6 (centeredRebase g6 : ExactMatrix Q4 N 1 6) := ⟨hQ4, rfl⟩
  have hg4u : centeredRebaseRuns g4 (centeredRebase g4 : ExactMatrix Q3 N 1 4) := ⟨hQ3, rfl⟩
  have hg2u : centeredRebaseRuns g2 (centeredRebase g2 : ExactMatrix Q2 N 1 2) := ⟨hQ2, rfl⟩
  -- Encryption and preprocessing.
  have hstate := encrypt_state henc hsm hsp hg12 hg12u
  rw [hμ0] at hstate
  have hmembers := encrypt_member henc hsm hG14
  have hmemα : ∀ k : Fin 17, Rel sm (memberKey model external.input_0 k)
      (fun j ↦ reducePoly Q7 N (α k * unitPair sm j) • G14)
      ![execution.«stage_3».2.2.1 k, execution.«stage_3».2.2.2.1 k] 26 := fun k ↦ by
    rw [← hα k]; exact (hmembers k).1
  have hpayα : ∀ k : Fin 17, execution.«stage_3».2.2.2.2.1 k = fun _ _ ↦ reducePoly Q7 N (α k) :=
    fun k ↦ by rw [← hα k]; exact (hmembers k).2
  have hhelp := encrypt_helper5 henc hsm hg2 hg2u
  have hrhs0 := preprocess_selector0 hpre hsp hg10 hg10u
  have hrhs1 := preprocess_selector1 hpre hsp hg8 hg8u
  have hrhs2 := preprocess_selector2 hpre hsp hg6 hg6u
  have hrhs3 := preprocess_selector3 hpre hsp hg4 hg4u
  have hrhs4 := preprocess_selector4 hpre hsp hg2 hg2u
  have hsw := preprocess_switch hpre hsp
  -- The five blocks accumulate the monomial of the sum of the selected coordinates.
  obtain ⟨w, hb⟩ := honline
  -- Block 0.
  have hb0 := block0 hb hsmn (μ := Xe 0) (α := α) (ν := fun i ↦ message (n := N) (s0 i) (x0 i))
    (fun i ↦ by rw [Xe_mul]; exact polyNorm_Xe hN1 _) (fun k ↦ polyNorm_Xe hN1 _) hG14 hg12 hg12u hmemα
    hpayα hstate hrhs0
  have hin1 := hb0.target_eq (collapse _ _ _ _ _ _ sel0 hsel0 hν0)
  rw [Xe_mul] at hin1
  obtain ⟨z1, hz1⟩ : ∃ z : Nat, z = (a ⟨coords 0 sel0, coords_lt 0 sel0⟩).toNat + 0 := ⟨_, rfl⟩
  rw [show 2 * (a ⟨coords 0 sel0, coords_lt 0 sel0⟩).toNat + 0 = 2 * z1 by
    rw [hz1]; ring] at hin1
  -- Block 1.
  have hb1 := block1 hb hsmn (μ := Xe (2 * z1)) (α := α) (ν := fun i ↦ message (n := N) (s1 i) (x1 i))
    (fun i ↦ by rw [Xe_mul]; exact polyNorm_Xe hN1 _) (fun k ↦ polyNorm_Xe hN1 _) hG14 hg10 hg10u hmemα
    hpayα hin1 hrhs1
  have hin2 := hb1.target_eq (collapse _ _ _ _ _ _ sel1 hsel1 hν1)
  rw [Xe_mul] at hin2
  obtain ⟨z2, hz2⟩ : ∃ z : Nat, z = (a ⟨coords 1 sel1, coords_lt 1 sel1⟩).toNat + z1 := ⟨_, rfl⟩
  rw [show 2 * (a ⟨coords 1 sel1, coords_lt 1 sel1⟩).toNat + 2 * z1 = 2 * z2 by
    rw [hz2]; ring] at hin2
  -- Block 2.
  have hb2 := block2 hb hsmn (μ := Xe (2 * z2)) (α := α) (ν := fun i ↦ message (n := N) (s2 i) (x2 i))
    (fun i ↦ by rw [Xe_mul]; exact polyNorm_Xe hN1 _) (fun k ↦ polyNorm_Xe hN1 _) hG14 hg8 hg8u hmemα
    hpayα hin2 hrhs2
  have hin3 := hb2.target_eq (collapse _ _ _ _ _ _ sel2 hsel2 hν2)
  rw [Xe_mul] at hin3
  obtain ⟨z3, hz3⟩ : ∃ z : Nat, z = (a ⟨coords 2 sel2, coords_lt 2 sel2⟩).toNat + z2 := ⟨_, rfl⟩
  rw [show 2 * (a ⟨coords 2 sel2, coords_lt 2 sel2⟩).toNat + 2 * z2 = 2 * z3 by
    rw [hz3]; ring] at hin3
  -- Block 3.
  have hb3 := block3 hb hsmn (μ := Xe (2 * z3)) (α := α) (ν := fun i ↦ message (n := N) (s3 i) (x3 i))
    (fun i ↦ by rw [Xe_mul]; exact polyNorm_Xe hN1 _) (fun k ↦ polyNorm_Xe hN1 _) hG14 hg6 hg6u hmemα
    hpayα hin3 hrhs3
  have hin4 := hb3.target_eq (collapse _ _ _ _ _ _ sel3 hsel3 hν3)
  rw [Xe_mul] at hin4
  obtain ⟨z4, hz4⟩ : ∃ z : Nat, z = (a ⟨coords 3 sel3, coords_lt 3 sel3⟩).toNat + z3 := ⟨_, rfl⟩
  rw [show 2 * (a ⟨coords 3 sel3, coords_lt 3 sel3⟩).toNat + 2 * z3 = 2 * z4 by
    rw [hz4]; ring] at hin4
  -- Block 4.
  have hb4 := block4 hb hsmn (μ := Xe (2 * z4)) (α := α) (ν := fun i ↦ message (n := N) (s4 i) (x4 i))
    (fun i ↦ by rw [Xe_mul]; exact polyNorm_Xe hN1 _) (fun k ↦ polyNorm_Xe hN1 _) hG14 hg4 hg4u hmemα
    hpayα hin4 hrhs4
  have hin5 := hb4.target_eq (collapse _ _ _ _ _ _ sel4 hsel4 hν4)
  rw [Xe_mul] at hin5
  obtain ⟨z5, hz5⟩ : ∃ z : Nat, z = (a ⟨coords 4 sel4, coords_lt 4 sel4⟩).toNat + z4 := ⟨_, rfl⟩
  rw [show 2 * (a ⟨coords 4 sel4, coords_lt 4 sel4⟩).toNat + 2 * z4 = 2 * z5 by
    rw [hz5]; ring] at hin5
  -- The readout LUT and the decryption.
  have hlut := lutRound hb hsmn hg2 hg2u hin5 hhelp hsw
  have hb2 := hb
  unfold Stage_online.generatedRoot.body at hb2
  extract_lets at hb2
  have houtOnline := hb2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨wd, hd⟩ := hdec
  unfold Stage_decode.generatedRoot.body at hd
  dsimp only at hd
  obtain ⟨hm1, hm2, -, hdec1, hout⟩ := hd
  rw [hsm] at hm1
  have hw2 := modulusReduce_poly hQ7 hQ1 hN hm1
  have hw6 := modulusReduce_one hQ7 hQ1 hN hm2
  have hbound : 4 * lutBound (∑ _i : Fin 14, cellBound4 (∑ _i : Fin 14, cellBound3
      (∑ _i : Fin 14, cellBound2 (∑ _i : Fin 14, cellBound1 (∑ _i : Fin 14, cellBound0 26))))) + 8 < Q1 := by
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
    decide
  have hval := decode_terminal_parity hQ1 (by norm_num) hN hlut hbound (k := ⟨0, hN⟩)
    (lutValue_coeff_bound z5) (by
      rw [houtOnline, hw2, hw6]
      funext r c
      obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _
      simp only [matrixSub, matrixAdd, matrixMulScalarLeft_eq, Matrix.sub_apply, Matrix.add_apply,
        Matrix.smul_apply, smul_eq_mul, Matrix.one_apply_eq, one_mul]
      rfl) hdec1
  rw [hout, hval, readout_parity, hid]
  have hz : selectedSum a ![s0, s1, s2, s3, s4] = (z5 : Int) := by
    rw [selectedSum, Fin.sum_univ_five,
      show (![s0, s1, s2, s3, s4] : Fin 5 → Fin 14 → Int) 0 = s0 from rfl,
      show (![s0, s1, s2, s3, s4] : Fin 5 → Fin 14 → Int) 1 = s1 from rfl,
      show (![s0, s1, s2, s3, s4] : Fin 5 → Fin 14 → Int) 2 = s2 from rfl,
      show (![s0, s1, s2, s3, s4] : Fin 5 → Fin 14 → Int) 3 = s3 from rfl,
      show (![s0, s1, s2, s3, s4] : Fin 5 → Fin 14 → Int) 4 = s4 from rfl]
    rw [one_hot_sum a 0 hsel0, one_hot_sum a 1 hsel1, one_hot_sum a 2 hsel2, one_hot_sum a 3 hsel3,
      one_hot_sum a 4 hsel4, hz5, hz4, hz3, hz2, hz1]
    push_cast
    rw [Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1,
      Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1]
    ring
  rw [hz]

/-- The worst-case correctness claim of `worst-n32-crt7x40`. -/
theorem certificate : GeneratedClaim.CorrectnessClaim :=
  fun _ _ _ hruns ↦ correct hruns

/-- info: 'LwrProof.certificate' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms certificate

end LwrProof
