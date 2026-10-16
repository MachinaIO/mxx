import PeKeygenScatterBase

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxHeartbeats 2000000 in
theorem helperScatterStep {model : HashModel} {tape : SampleTape} {path : List Nat}
    {params : Stage_keygen.Params} {key : ByteArray} {row lane : Nat}
    (row_bound : row < 256) (lane_bound : lane < 45)
    {packed : ExactMatrix Q N 60 45} {current next : ExactMatrix Q N 1 256}
    (h : Stage_keygen.sequential_parallel_generatedRoot_14_5 Backend.backend model tape path
      params row lane (current, key, packed, gZero * packed, ()) next) :
    next = current + scatterTerm model key packed ⟨row, row_bound⟩ ⟨lane, lane_bound⟩ := by
  obtain ⟨key0, key1, joined, routed, positive, rowUnit, negative, destinationUnit, h0, h1⟩ := h
  obtain ⟨-, -, hash0, -, -, hash1, concat, -, -, route, -, -, -, -, -, -, positive_slice,
    -, -, -, -, -, -, -, -, row_slice, -, -, -, -, -, -⟩ := h0
  obtain ⟨negative_slice, -, -, -, -, -, -, -, -, -, -, -, -, destination_slice, output⟩ := h1
  have helper_cast := kHelp_cast lane lane_bound
  rw [← helper_cast] at hash0 hash1
  have key0_eq : key0 = helperKey model key (kHelp lane) 0 0 :=
    helperKey_read (c := 0) (by norm_num) hash0
  have key1_eq : key1 = helperKey model key (kHelp lane) 0 1 :=
    helperKey_read (c := 1) (by norm_num) hash1
  rw [key0_eq, key1_eq] at concat
  have joined_eq := concat_joinRow (r := helperKey model key (kHelp lane) 0) concat
  have routed_eq := routing_slot row_bound route
  rw [joined_eq] at routed_eq
  obtain ⟨_, positive_eq⟩ := sliceMatrix_column positive_slice
  obtain ⟨_, negative_eq⟩ := sliceMatrix_column negative_slice
  have row_unit := unitRow_window (c := (row : Int))
    (by simp only [Int.ofNat_eq_natCast]; ring) row_slice
  have destination_unit := unitRow_window
    (c := MxxIR.selectInt (45 * Int.ofNat row + Int.ofNat lane) scatterList)
    (by unfold scatterList; ring) destination_slice
  rw [scatterList_read row lane row_bound lane_bound] at destination_unit
  rw [output]
  funext entry target
  rw [Subsingleton.elim entry 0]
  change current 0 target + positive 0 0 * rowUnit 0 target +
    (0 - negative 0 0) * destinationUnit 0 target = _
  rw [positive_eq, negative_eq, row_unit, destination_unit, routed_eq]
  simp only [matrixMul_eq_mul, scatterTerm, unitRow, Matrix.add_apply,
    Int.ofNat_eq_natCast, Int.toNat_natCast, Nat.cast_inj, Fin.ext_iff]
  split_ifs <;> simp only [mul_one, mul_zero] <;> abel

set_option maxHeartbeats 2000000 in
theorem helperScatterRow {model : HashModel} {tape : SampleTape} {path : List Nat}
    {params : Stage_keygen.Params} {key : ByteArray} {row : Nat} (row_bound : row < 256)
    {packed : ExactMatrix Q N 60 45} {output : ExactMatrix Q N 1 256}
    (h : Stage_keygen.parallel_generatedRoot_14 Backend.backend model tape path params row
      (key, packed, gZero, ()) output) :
    output = ∑ lane : Fin 45, scatterTerm model key packed ⟨row, row_bound⟩ lane := by
  obtain ⟨witness, -, loop, output_eq⟩ := h
  have total := stream_sum _ (fun lane current next bound body ↦
    helperScatterStep row_bound bound body) loop
  rw [output_eq]
  simpa only [zero_add] using total

theorem scatter_gather (model : HashModel) (key : ByteArray)
    (packed : Fin 256 → ExactMatrix Q N 60 45) (target : Fin 256) :
    (∑ row : Fin 256, ∑ lane : Fin 45, scatterTerm model key (packed row) row lane) 0 target =
      (∑ lane : Fin 45, helperTerm model key (packedHelpers packed) target lane) 0 0 := by
  classical
  simp only [Matrix.sum_apply]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun lane _ ↦ ?_
  let paired : Fin 256 := ⟨pairRow (45 * target + lane), pairRow_lt _⟩
  have select : ∀ row : Fin 256,
      target.val = (row.val + helperIdx lane) % 256 ↔ row = paired := by
    intro row
    have bound := helperIdx_lt lane lane.isLt
    have equal := pairRow_eq target lane target.isLt lane.isLt
    simp only [Fin.ext_iff, paired]
    omega
  have contribution : ∀ row : Fin 256,
      scatterTerm model key (packed row) row lane 0 target =
        (if row = target then
          (substituteMatrix (slotAut target) (slotAut_odd target)
            (joinRow (helperKey model key (kHelp lane) 0)) * packed target) 0 lane else 0) -
        (if row = paired then (gZero * packed paired) 0 lane else 0) := by
    intro row
    unfold scatterTerm
    simp only [select row]
    congr 1
    · by_cases equal : row = target
      · subst row
        rfl
      · simp only [equal, Ne.symm equal, if_false]
    · by_cases equal : row = paired
      · subst row
        rfl
      · simp only [equal, if_false]
  simp only [contribution, Finset.sum_sub_distrib, Finset.sum_ite_eq', Finset.mem_univ,
    if_true]
  simp only [helperTerm, Matrix.sub_apply]
  rw [packedHelpers_entry]
  have pair_eq : (⟨lane.val + 45 * pairRow (45 * target + lane), by
      have := pairRow_lt (45 * target + lane)
      omega⟩ : Fin 11520) = ⟨45 * paired.val + lane.val, by omega⟩ := by
    ext
    dsimp [paired]
    omega
  rw [pair_eq, packedHelpers_entry]
  rfl

end PeProof
