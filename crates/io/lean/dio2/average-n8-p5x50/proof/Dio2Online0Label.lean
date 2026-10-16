import Dio2Label
import Dio2Online0Refresh

/-!
# The label targets of step 0

The refresh parts of the label pairs of step 0 (`dio2/label` members `0 .. 3`, values `α_i`,
payloads `w_391`) re-target label products to `E^σ = (w_244, w_245)` (`label_finish`).
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen RefreshTopProof.GenFinal
  W_Stage_online_0

/-- `rns-bgg/root-input/dio2/label`, the tag prefix of the label inputs. -/
abbrev labelInPre : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100,
    105, 111, 50, 47, 108, 97, 98, 101, 108]

theorem familyGetStatic_fin {q m W : Nat} {f : Fin W → ExactMatrix q 8 1 m} {x : ExactMatrix q 8 1 m}
    (i : Fin W) (h : familyGetStatic f (i : Nat) x) : x = f i := by
  obtain ⟨pos, hpos, rfl⟩ := h
  congr 1
  exact Fin.ext (by omega)

/-- A label member `i` (tag prefix `pre`, rows of the families `f₀, f₁`) as an encoding relation
with its hashed key blocks. -/
theorem label_rel_of_hash {q W : Nat} {model : HashModel} {key : ByteArray} {pre : Blob}
    {sm : ErrorPoly 8} {t : Fin 2 → ExactMatrix q 8 1 25} {Ba : Nat}
    {f0 f1 : Fin W → ExactMatrix q 8 1 25} (i : Fin W)
    (hlab : Rel sm (labelKeysG model key q 25 pre (i : Nat)) t ![f0 i, f1 i] Ba)
    {k00 k01 k10 k11 c0 c1 : ExactMatrix q 8 1 25}
    (h00 : hashSample model pre [.bytes memberTagG, .u64Le (i : Nat), .bytes componentTagG,
      .u64Le 0] key k00)
    (h01 : hashSample model pre [.bytes memberTagG, .u64Le (i : Nat), .bytes componentTagG,
      .u64Le 1] key k01)
    (h10 : hashSample model pre [.bytes memberTagG, .u64Le (i : Nat), .bytes componentTagG,
      .u64Le 2] key k10)
    (h11 : hashSample model pre [.bytes memberTagG, .u64Le (i : Nat), .bytes componentTagG,
      .u64Le 3] key k11)
    (hc0 : familyGetStatic f0 (i : Nat) c0) (hc1 : familyGetStatic f1 (i : Nat) c1) :
    Rel sm ![![k00, k01], ![k10, k11]] t ![c0, c1] Ba := by
  have e00 := keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le (i : Nat)]) 0 0 (by norm_num) h00
  have e01 := keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le (i : Nat)]) 0 1 (by norm_num) h01
  have e10 := keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le (i : Nat)]) 1 0 (by norm_num) h10
  have e11 := keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le (i : Nat)]) 1 1 (by norm_num) h11
  have e : labelKeysG model key q 25 pre (i : Nat) = ![![k00, k01], ![k10, k11]] := by
    funext a j
    fin_cases a <;> fin_cases j
    exacts [e00.symm, e01.symm, e10.symm, e11.symm]
  rw [e, ← familyGetStatic_fin i hc0, ← familyGetStatic_fin i hc1] at hlab
  exact hlab

set_option maxHeartbeats 32000000 in
/-- **The label target `(w_425, w_438)` of step 0**: label members `1` and `3`. -/
theorem online0_label_425 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm : ErrorPoly 8} {α : Fin 4 → ErrorPoly 8} (hα : ∀ i, polyNorm (α i) ≤ 1)
    (hα13 : polyNorm (α 1 * α 3) ≤ 1) {Ba : Nat}
    (hlab : ∀ i : Fin 4, Rel sm
      (labelKeysG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 labelInPre (i : Nat))
      (fun j ↦ reducePoly modulus_0 8 (α i * unitPair sm j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_372_0 ctx i, generatedRoot.w_377_0 ctx i] Ba)
    (hpay : ∀ i : Fin 4, generatedRoot.w_391_0 ctx i = fun _ _ ↦ reducePoly modulus_0 8 (α i))
    {ts : Fin 2 → ExactMatrix modulus_0 8 1 25} {Bs : Nat}
    (hs : Rel sm ![![generatedRoot.w_402_0 ctx, generatedRoot.w_428_0 ctx],
        ![generatedRoot.w_417_0 ctx, generatedRoot.w_433_0 ctx]] ts
      ![generatedRoot.w_244_0 ctx, generatedRoot.w_245_0 ctx] Bs) :
    Rel sm (fun k j ↦
        rowOf (matrixMul (generatedRoot.w_382_0 ctx) (generatedRoot.w_387_0 ctx)) k.succ *
            generatedRoot.w_399_0 ctx * ![generatedRoot.w_403_0 ctx, generatedRoot.w_429_0 ctx] j +
          rowOf (matrixMul (generatedRoot.w_382_0 ctx) (generatedRoot.w_409_0 ctx)) k.succ *
            generatedRoot.w_399_0 ctx * ![generatedRoot.w_418_0 ctx, generatedRoot.w_434_0 ctx] j)
      (fun j ↦ reducePoly modulus_0 8 (α 1 * α 3) • ts j)
      ![generatedRoot.w_425_0 ctx, generatedRoot.w_438_0 ctx]
      (25 * 8 * (25 * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Ba) * (4096 / 2)) * (4096 / 2) +
        25 * 8 * (25 * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Ba) * (4096 / 2)) * (4096 / 2) +
        8 * 1 * Bs) := by
  have hG := F.c1.h57_0
  have ha := label_rel_of_hash 1 (hlab 1) F.c13.h374_0 F.c14.h379_0 F.c13.h375_0 F.c14.h380_0
    F.c13.h373_0 F.c14.h378_0
  have hb := label_rel_of_hash 3 (hlab 3) F.c14.h383_0 F.c14.h405_0 F.c14.h384_0 F.c14.h406_0
    F.c14.h390_0 F.c15.h412_0
  have hp1 : generatedRoot.w_392_0 ctx = fun _ _ ↦ reducePoly modulus_0 8 (α 1) := by
    obtain ⟨pos, hpos, h⟩ := F.c14.h392_0
    rw [h, ← hpay 1]; congr 1; exact Fin.ext (by omega)
  have hp3 : generatedRoot.w_421_0 ctx = fun _ _ ↦ reducePoly modulus_0 8 (α 3) := by
    obtain ⟨pos, hpos, h⟩ := F.c15.h421_0
    rw [h, ← hpay 3]; congr 1; exact Fin.ext (by omega)
  have hprod := label_last (by norm_num) hq1 layout hlayout hexact hG (hα 1) ha hb
    F.c14.h374_0_1 F.c14.h376_0 F.c14.h379_0_1 F.c14.h381_0 F.c14.h382_0
    (Xs := ![generatedRoot.w_385_0 ctx, generatedRoot.w_407_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c14.h385_0, F.c14.h407_0])
    (D' := ![generatedRoot.w_386_0 ctx, generatedRoot.w_408_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c14.h386_0, F.c14.h408_0])
    (D := ![generatedRoot.w_387_0 ctx, generatedRoot.w_409_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c14.h387_0, F.c14.h409_0]) hp1
    (Y := ![generatedRoot.w_389_0 ctx, generatedRoot.w_411_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c14.h389_0, F.c15.h411_0])
  have hX : generatedRoot.w_397_0 ctx = generatedRoot.w_57_0 ctx :=
    gadget_rebase_one hq1 F.c14.h395_0
  have hDg : generatedRoot.w_399_0 ctx = generatedRoot.w_398_0 ctx :=
    compactRebase_self_gen (by norm_num) hq1 F.c14.h399_0
  have h398 := F.c14.h398_0
  rw [hX, ← hDg] at h398
  have hfin := label_finish (by norm_num) hq1 layout hlayout hexact hG hα13 hprod h398 hs
    (D0 := ![generatedRoot.w_403_0 ctx, generatedRoot.w_429_0 ctx])
    (D1 := ![generatedRoot.w_418_0 ctx, generatedRoot.w_434_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c14.h403_0, F.c15.h429_0])
    (fun j ↦ by fin_cases j; exacts [F.c15.h418_0, F.c15.h434_0])
  have h423 : generatedRoot.w_423_0 ctx = fun _ _ ↦ reducePoly modulus_0 8 (α 1 * α 3) := by
    rw [modulusReduce_same (by norm_num) F.c15.h423_0]
    exact payload_mul hp1 hp3
  have hc : (fun j ↦ (![generatedRoot.w_389_0 ctx, generatedRoot.w_411_0 ctx] 0 +
        matrixMulScalarRight (![generatedRoot.w_390_0 ctx, generatedRoot.w_412_0 ctx] 0)
          (generatedRoot.w_392_0 ctx)) * generatedRoot.w_399_0 ctx *
        ![generatedRoot.w_403_0 ctx, generatedRoot.w_429_0 ctx] j +
      (![generatedRoot.w_389_0 ctx, generatedRoot.w_411_0 ctx] 1 +
        matrixMulScalarRight (![generatedRoot.w_390_0 ctx, generatedRoot.w_412_0 ctx] 1)
          (generatedRoot.w_392_0 ctx)) * generatedRoot.w_399_0 ctx *
        ![generatedRoot.w_418_0 ctx, generatedRoot.w_434_0 ctx] j +
      reducePoly modulus_0 8 (α 1 * α 3) • ![generatedRoot.w_244_0 ctx, generatedRoot.w_245_0 ctx] j) =
      ![generatedRoot.w_425_0 ctx, generatedRoot.w_438_0 ctx] := by
    funext j
    fin_cases j
    · show _ = matrixAdd (matrixAdd (matrixMul (matrixMul (generatedRoot.w_394_0 ctx)
          (generatedRoot.w_399_0 ctx)) (generatedRoot.w_403_0 ctx)) (matrixMul (matrixMul
          (generatedRoot.w_414_0 ctx) (generatedRoot.w_399_0 ctx)) (generatedRoot.w_418_0 ctx)))
          (matrixMulScalarLeft (generatedRoot.w_423_0 ctx) (generatedRoot.w_244_0 ctx))
      rw [matrixMulScalarLeft_eq, h423]
      rfl
    · show _ = matrixAdd (matrixAdd (matrixMul (matrixMul (generatedRoot.w_394_0 ctx)
          (generatedRoot.w_399_0 ctx)) (generatedRoot.w_429_0 ctx)) (matrixMul (matrixMul
          (generatedRoot.w_414_0 ctx) (generatedRoot.w_399_0 ctx)) (generatedRoot.w_434_0 ctx)))
          (matrixMulScalarLeft (generatedRoot.w_423_0 ctx) (generatedRoot.w_245_0 ctx))
      rw [matrixMulScalarLeft_eq, h423]
      rfl
  rw [← hc]
  exact hfin

set_option maxHeartbeats 64000000 in
/-- **The label target `(w_543, w_550)` of step 0**: `(α₀ α₁) (α₂ α₃)`. -/
theorem online0_label_543 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm : ErrorPoly 8} {α : Fin 4 → ErrorPoly 8} (hα : ∀ i, polyNorm (α i) ≤ 1)
    (hα01 : polyNorm (α 0 * α 1) ≤ 1) (hα0123 : polyNorm (α 0 * α 1 * (α 2 * α 3)) ≤ 1) {Ba : Nat}
    (hlab : ∀ i : Fin 4, Rel sm
      (labelKeysG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 labelInPre (i : Nat))
      (fun j ↦ reducePoly modulus_0 8 (α i * unitPair sm j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_372_0 ctx i, generatedRoot.w_377_0 ctx i] Ba)
    (hpay : ∀ i : Fin 4, generatedRoot.w_391_0 ctx i = fun _ _ ↦ reducePoly modulus_0 8 (α i))
    {ts : Fin 2 → ExactMatrix modulus_0 8 1 25} {Bs : Nat}
    (hs : Rel sm ![![generatedRoot.w_402_0 ctx, generatedRoot.w_428_0 ctx],
        ![generatedRoot.w_417_0 ctx, generatedRoot.w_433_0 ctx]] ts
      ![generatedRoot.w_244_0 ctx, generatedRoot.w_245_0 ctx] Bs) :
    Rel sm (fun k j ↦
        rowOf (matrixMul (generatedRoot.w_475_0 ctx) (generatedRoot.w_495_0 ctx)) k.succ *
            generatedRoot.w_511_0 ctx * ![generatedRoot.w_513_0 ctx, generatedRoot.w_544_0 ctx] j +
          rowOf (matrixMul (generatedRoot.w_475_0 ctx) (generatedRoot.w_525_0 ctx)) k.succ *
            generatedRoot.w_511_0 ctx * ![generatedRoot.w_535_0 ctx, generatedRoot.w_546_0 ctx] j)
      (fun j ↦ reducePoly modulus_0 8 (α 0 * α 1 * (α 2 * α 3)) • ts j)
      ![generatedRoot.w_543_0 ctx, generatedRoot.w_550_0 ctx]
      (25 * 8 * (25 * 8 * ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Ba) * 2048 +
          8 * 1 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Ba)) * (4096 / 2)) * (4096 / 2) +
        25 * 8 * (25 * 8 * ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Ba) * 2048 +
          8 * 1 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Ba)) * (4096 / 2)) * (4096 / 2) +
        8 * 1 * Bs) := by
  have hG := F.c1.h57_0
  have hm0 := label_rel_of_hash 0 (hlab 0) F.c15.h440_0 F.c15.h444_0 F.c15.h441_0 F.c15.h445_0
    F.c15.h439_0 F.c15.h443_0
  have hm1 := label_rel_of_hash 1 (hlab 1) F.c15.h448_0 F.c16.h462_0 F.c15.h449_0 F.c16.h463_0
    F.c16.h455_0 F.c17.h469_0
  have hm2 := label_rel_of_hash 2 (hlab 2) F.c17.h477_0 F.c17.h481_0 F.c17.h478_0 F.c17.h482_0
    F.c17.h476_0 F.c17.h480_0
  have hm3 := label_rel_of_hash 3 (hlab 3) F.c18.h485_0 F.c19.h515_0 F.c18.h486_0 F.c19.h516_0
    F.c19.h499_0 F.c20.h529_0
  have hp : ∀ (i : Fin 4) {x : ExactMatrix modulus_0 8 1 1},
      familyGetStatic (generatedRoot.w_391_0 ctx) (i : Nat) x →
      x = fun _ _ ↦ reducePoly modulus_0 8 (α i) := fun i _ h ↦ by
    rw [familyGetStatic_fin i h, hpay i]
  have hp0 := hp 0 F.c16.h456_0
  have hp1 := hp 1 F.c19.h503_0
  have hp2 := hp 2 F.c19.h500_0
  have hp3 := hp 3 F.c20.h538_0
  have hP1 := label_mul (by norm_num) hq1 layout hlayout hexact hG (hα 0) hm0 hm1
    F.c15.h440_0_1 F.c15.h442_0 F.c15.h444_0_1 F.c15.h446_0 F.c15.h447_0
    (Xs := ![generatedRoot.w_450_0 ctx, generatedRoot.w_464_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c15.h450_0, F.c16.h464_0])
    (D' := ![generatedRoot.w_451_0 ctx, generatedRoot.w_465_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c15.h451_0, F.c16.h465_0])
    (D := ![generatedRoot.w_452_0 ctx, generatedRoot.w_466_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c15.h452_0, F.c16.h466_0]) hp0
    (Y := ![generatedRoot.w_454_0 ctx, generatedRoot.w_468_0 ctx])
    (A' := ![![generatedRoot.w_459_0 ctx, generatedRoot.w_472_0 ctx],
      ![generatedRoot.w_460_0 ctx, generatedRoot.w_473_0 ctx]])
    (fun j ↦ by fin_cases j; exacts [F.c16.h454_0, F.c17.h468_0])
    (fun j ↦ by fin_cases j; exacts [F.c16.h459_0, F.c17.h472_0])
    (fun j ↦ by fin_cases j; exacts [F.c16.h460_0, F.c17.h473_0])
  have hP2 := label_mul (by norm_num) hq1 layout hlayout hexact hG (hα 2) hm2 hm3
    F.c17.h477_0_1 F.c17.h479_0 F.c17.h481_0_1 F.c17.h483_0 F.c17.h484_0
    (Xs := ![generatedRoot.w_487_0 ctx, generatedRoot.w_517_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c18.h487_0, F.c19.h517_0])
    (D' := ![generatedRoot.w_488_0 ctx, generatedRoot.w_518_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c18.h488_0, F.c19.h518_0])
    (D := ![generatedRoot.w_489_0 ctx, generatedRoot.w_519_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c18.h489_0, F.c19.h519_0]) hp2
    (Y := ![generatedRoot.w_498_0 ctx, generatedRoot.w_528_0 ctx])
    (A' := ![![generatedRoot.w_491_0 ctx, generatedRoot.w_521_0 ctx],
      ![generatedRoot.w_492_0 ctx, generatedRoot.w_522_0 ctx]])
    (fun j ↦ by fin_cases j; exacts [F.c19.h498_0, F.c20.h528_0])
    (fun j ↦ by fin_cases j; exacts [F.c18.h491_0, F.c19.h521_0])
    (fun j ↦ by fin_cases j; exacts [F.c18.h492_0, F.c19.h522_0])
  have hP1' : Rel sm ![![generatedRoot.w_459_0 ctx, generatedRoot.w_472_0 ctx],
      ![generatedRoot.w_460_0 ctx, generatedRoot.w_473_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (α 0 * α 1 * unitPair sm j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_458_0 ctx, generatedRoot.w_471_0 ctx] ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Ba) := by
    convert hP1 using 1
    funext j; fin_cases j <;> rfl
  have hP2' : Rel sm ![![generatedRoot.w_491_0 ctx, generatedRoot.w_521_0 ctx],
      ![generatedRoot.w_492_0 ctx, generatedRoot.w_522_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (α 2 * α 3 * unitPair sm j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_502_0 ctx, generatedRoot.w_531_0 ctx] ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Ba) := by
    convert hP2 using 1
    funext j; fin_cases j <;> rfl
  have hp01 := payload_mul hp0 hp1
  have hprod := label_last (by norm_num) hq1 layout hlayout hexact hG hα01 hP1' hP2'
    F.c16.h459_0_1 F.c16.h461_0 F.c17.h472_0_1 F.c17.h474_0 F.c17.h475_0
    (Xs := ![generatedRoot.w_493_0 ctx, generatedRoot.w_523_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c18.h493_0, F.c19.h523_0])
    (D' := ![generatedRoot.w_494_0 ctx, generatedRoot.w_524_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c18.h494_0, F.c19.h524_0])
    (D := ![generatedRoot.w_495_0 ctx, generatedRoot.w_525_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c18.h495_0, F.c20.h525_0]) (payload := generatedRoot.w_504_0 ctx)
    hp01
    (Y := ![generatedRoot.w_497_0 ctx, generatedRoot.w_527_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c18.h497_0, F.c20.h527_0])
  have hX : generatedRoot.w_509_0 ctx = generatedRoot.w_57_0 ctx :=
    gadget_rebase_one hq1 F.c19.h507_0
  have hDg : generatedRoot.w_511_0 ctx = generatedRoot.w_510_0 ctx :=
    compactRebase_self_gen (by norm_num) hq1 F.c19.h511_0
  have h510 := F.c19.h510_0
  rw [hX, ← hDg] at h510
  have hfin := label_finish (by norm_num) hq1 layout hlayout hexact hG hα0123 hprod h510 hs
    (D0 := ![generatedRoot.w_513_0 ctx, generatedRoot.w_544_0 ctx])
    (D1 := ![generatedRoot.w_535_0 ctx, generatedRoot.w_546_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c19.h513_0, F.c20.h544_0])
    (fun j ↦ by fin_cases j; exacts [F.c20.h535_0, F.c20.h546_0])
  have h541 : generatedRoot.w_541_0 ctx =
      fun _ _ ↦ reducePoly modulus_0 8 (α 0 * α 1 * (α 2 * α 3)) := by
    rw [modulusReduce_same (by norm_num) F.c20.h541_0]
    exact payload_mul hp01 (payload_mul hp2 hp3)
  refine Rel.c_eq hfin ?_
  funext j
  fin_cases j
  · show _ = matrixAdd (matrixAdd (matrixMul (matrixMul (generatedRoot.w_506_0 ctx)
        (generatedRoot.w_511_0 ctx)) (generatedRoot.w_513_0 ctx)) (matrixMul (matrixMul
        (generatedRoot.w_533_0 ctx) (generatedRoot.w_511_0 ctx)) (generatedRoot.w_535_0 ctx)))
        (matrixMulScalarLeft (generatedRoot.w_541_0 ctx) (generatedRoot.w_244_0 ctx))
    rw [matrixMulScalarLeft_eq, h541]
    rfl
  · show _ = matrixAdd (matrixAdd (matrixMul (matrixMul (generatedRoot.w_506_0 ctx)
        (generatedRoot.w_511_0 ctx)) (generatedRoot.w_544_0 ctx)) (matrixMul (matrixMul
        (generatedRoot.w_533_0 ctx) (generatedRoot.w_511_0 ctx)) (generatedRoot.w_546_0 ctx)))
        (matrixMulScalarLeft (generatedRoot.w_541_0 ctx) (generatedRoot.w_245_0 ctx))
    rw [matrixMulScalarLeft_eq, h541]
    rfl

set_option maxHeartbeats 64000000 in
/-- **The label target `(w_734, w_741)` of step 0**: `α₁ α₂`. -/
theorem online0_label_734 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm : ErrorPoly 8} {α : Fin 4 → ErrorPoly 8} (hα : ∀ i, polyNorm (α i) ≤ 1) {Ba : Nat}
    (hlab : ∀ i : Fin 4, Rel sm
      (labelKeysG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 labelInPre (i : Nat))
      (fun j ↦ reducePoly modulus_0 8 (α i * unitPair sm j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_372_0 ctx i, generatedRoot.w_377_0 ctx i] Ba)
    (hpay : ∀ i : Fin 4, generatedRoot.w_391_0 ctx i = fun _ _ ↦ reducePoly modulus_0 8 (α i))
    {ts : Fin 2 → ExactMatrix modulus_0 8 1 25} {Bs : Nat}
    (hs : Rel sm ![![generatedRoot.w_402_0 ctx, generatedRoot.w_428_0 ctx],
        ![generatedRoot.w_417_0 ctx, generatedRoot.w_433_0 ctx]] ts
      ![generatedRoot.w_244_0 ctx, generatedRoot.w_245_0 ctx] Bs)
    (hα12 : polyNorm (α 1 * α 2) ≤ 1) :
    Rel sm (fun k j ↦
        rowOf (matrixMul (generatedRoot.w_696_0 ctx) (generatedRoot.w_701_0 ctx)) k.succ *
            generatedRoot.w_712_0 ctx * ![generatedRoot.w_714_0 ctx, generatedRoot.w_735_0 ctx] j +
          rowOf (matrixMul (generatedRoot.w_696_0 ctx) (generatedRoot.w_720_0 ctx)) k.succ *
            generatedRoot.w_712_0 ctx * ![generatedRoot.w_727_0 ctx, generatedRoot.w_737_0 ctx] j)
      (fun j ↦ reducePoly modulus_0 8 (α 1 * α 2) • ts j)
      ![generatedRoot.w_734_0 ctx, generatedRoot.w_741_0 ctx]
      (25 * 8 * (25 * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Ba) * (4096 / 2)) * (4096 / 2) +
        25 * 8 * (25 * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Ba) * (4096 / 2)) * (4096 / 2) + 8 * 1 * Bs) := by
  have hG := F.c1.h57_0
  have hp : ∀ (i : Fin 4) {x : ExactMatrix modulus_0 8 1 1},
      familyGetStatic (generatedRoot.w_391_0 ctx) (i : Nat) x →
      x = fun _ _ ↦ reducePoly modulus_0 8 (α i) := fun i _ h ↦ by
    rw [familyGetStatic_fin i h, hpay i]
  have hm1 := label_rel_of_hash 1 (hlab 1) F.c25.h689_0 F.c25.h693_0 F.c25.h690_0 F.c25.h694_0
    F.c25.h688_0 F.c25.h692_0
  have hm2 := label_rel_of_hash 2 (hlab 2) F.c25.h697_0 F.c26.h716_0 F.c25.h698_0 F.c26.h717_0
    F.c26.h704_0 F.c26.h723_0
  have hp1 := hp 1 F.c26.h705_0
  have hp2 := hp 2 F.c26.h730_0
  have hprod := label_last (by norm_num) hq1 layout hlayout hexact hG (hα 1) hm1 hm2
    F.c25.h689_0_1 F.c25.h691_0 F.c25.h693_0_1 F.c25.h695_0 F.c25.h696_0
    (Xs := ![generatedRoot.w_699_0 ctx, generatedRoot.w_718_0 ctx]) (fun j ↦ by fin_cases j; exacts [F.c25.h699_0, F.c26.h718_0])
    (D' := ![generatedRoot.w_700_0 ctx, generatedRoot.w_719_0 ctx]) (fun j ↦ by fin_cases j; exacts [F.c25.h700_0, F.c26.h719_0])
    (D := ![generatedRoot.w_701_0 ctx, generatedRoot.w_720_0 ctx]) (fun j ↦ by fin_cases j; exacts [F.c25.h701_0, F.c26.h720_0]) (payload := generatedRoot.w_705_0 ctx) hp1
    (Y := ![generatedRoot.w_703_0 ctx, generatedRoot.w_722_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c26.h703_0, F.c26.h722_0])
  have hX : generatedRoot.w_710_0 ctx = generatedRoot.w_57_0 ctx :=
    gadget_rebase_one hq1 F.c26.h708_0
  have hDg : generatedRoot.w_712_0 ctx = generatedRoot.w_711_0 ctx :=
    compactRebase_self_gen (by norm_num) hq1 F.c26.h712_0
  have hdec := F.c26.h711_0
  rw [hX, ← hDg] at hdec
  have hfin := label_finish (by norm_num) hq1 layout hlayout hexact hG hα12 hprod hdec hs
    (D0 := ![generatedRoot.w_714_0 ctx, generatedRoot.w_735_0 ctx]) (D1 := ![generatedRoot.w_727_0 ctx, generatedRoot.w_737_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c26.h714_0, F.c26.h735_0])
    (fun j ↦ by fin_cases j; exacts [F.c26.h727_0, F.c26.h737_0])
  have hred : generatedRoot.w_732_0 ctx = fun _ _ ↦ reducePoly modulus_0 8 (α 1 * α 2) := by
    rw [modulusReduce_same (by norm_num) F.c26.h732_0]
    exact payload_mul hp1 hp2
  refine Rel.c_eq hfin ?_
  funext j
  fin_cases j
  · show _ = matrixAdd (matrixAdd (matrixMul (matrixMul (generatedRoot.w_707_0 ctx)
        (generatedRoot.w_712_0 ctx)) (generatedRoot.w_714_0 ctx)) (matrixMul (matrixMul
        (generatedRoot.w_725_0 ctx) (generatedRoot.w_712_0 ctx)) (generatedRoot.w_727_0 ctx)))
        (matrixMulScalarLeft (generatedRoot.w_732_0 ctx) (generatedRoot.w_244_0 ctx))
    rw [matrixMulScalarLeft_eq, hred]
    rfl
  · show _ = matrixAdd (matrixAdd (matrixMul (matrixMul (generatedRoot.w_707_0 ctx)
        (generatedRoot.w_712_0 ctx)) (generatedRoot.w_735_0 ctx)) (matrixMul (matrixMul
        (generatedRoot.w_725_0 ctx) (generatedRoot.w_712_0 ctx)) (generatedRoot.w_737_0 ctx)))
        (matrixMulScalarLeft (generatedRoot.w_732_0 ctx) (generatedRoot.w_245_0 ctx))
    rw [matrixMulScalarLeft_eq, hred]
    rfl

set_option maxHeartbeats 64000000 in
/-- **The label target `(w_626, w_633)` of step 0**: `(α₀ α₂) α₃`. -/
theorem online0_label_626 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm : ErrorPoly 8} {α : Fin 4 → ErrorPoly 8} (hα : ∀ i, polyNorm (α i) ≤ 1) {Ba : Nat}
    (hlab : ∀ i : Fin 4, Rel sm
      (labelKeysG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 labelInPre (i : Nat))
      (fun j ↦ reducePoly modulus_0 8 (α i * unitPair sm j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_372_0 ctx i, generatedRoot.w_377_0 ctx i] Ba)
    (hpay : ∀ i : Fin 4, generatedRoot.w_391_0 ctx i = fun _ _ ↦ reducePoly modulus_0 8 (α i))
    {ts : Fin 2 → ExactMatrix modulus_0 8 1 25} {Bs : Nat}
    (hs : Rel sm ![![generatedRoot.w_402_0 ctx, generatedRoot.w_428_0 ctx],
        ![generatedRoot.w_417_0 ctx, generatedRoot.w_433_0 ctx]] ts
      ![generatedRoot.w_244_0 ctx, generatedRoot.w_245_0 ctx] Bs)
    (hα02 : polyNorm (α 0 * α 2) ≤ 1) (hα023 : polyNorm (α 0 * α 2 * α 3) ≤ 1) :
    Rel sm (fun k j ↦
        rowOf (matrixMul (generatedRoot.w_587_0 ctx) (generatedRoot.w_592_0 ctx)) k.succ *
            generatedRoot.w_604_0 ctx * ![generatedRoot.w_606_0 ctx, generatedRoot.w_627_0 ctx] j +
          rowOf (matrixMul (generatedRoot.w_587_0 ctx) (generatedRoot.w_612_0 ctx)) k.succ *
            generatedRoot.w_604_0 ctx * ![generatedRoot.w_619_0 ctx, generatedRoot.w_629_0 ctx] j)
      (fun j ↦ reducePoly modulus_0 8 (α 0 * α 2 * α 3) • ts j)
      ![generatedRoot.w_626_0 ctx, generatedRoot.w_633_0 ctx]
      (25 * 8 * (25 * 8 * ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Ba) * 2048 + 8 * 1 * Ba) * (4096 / 2)) * (4096 / 2) +
        25 * 8 * (25 * 8 * ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Ba) * 2048 + 8 * 1 * Ba) * (4096 / 2)) * (4096 / 2) + 8 * 1 * Bs) := by
  have hG := F.c1.h57_0
  have hp : ∀ (i : Fin 4) {x : ExactMatrix modulus_0 8 1 1},
      familyGetStatic (generatedRoot.w_391_0 ctx) (i : Nat) x →
      x = fun _ _ ↦ reducePoly modulus_0 8 (α i) := fun i _ h ↦ by
    rw [familyGetStatic_fin i h, hpay i]
  have hm0 := label_rel_of_hash 0 (hlab 0) F.c20.h552_0 F.c20.h556_0 F.c20.h553_0 F.c20.h557_0
    F.c20.h551_0 F.c20.h555_0
  have hm2 := label_rel_of_hash 2 (hlab 2) F.c21.h560_0 F.c21.h574_0 F.c21.h561_0 F.c21.h575_0
    F.c21.h567_0 F.c22.h581_0
  have hm3 := label_rel_of_hash 3 (hlab 3) F.c22.h588_0 F.c23.h608_0 F.c22.h589_0 F.c23.h609_0
    F.c23.h595_0 F.c23.h615_0
  have hp0 := hp 0 F.c21.h568_0
  have hp2 := hp 2 F.c23.h596_0
  have hp3 := hp 3 F.c23.h622_0
  have hP1 := label_mul (by norm_num) hq1 layout hlayout hexact hG (hα 0) hm0 hm2
    F.c20.h552_0_1 F.c20.h554_0 F.c20.h556_0_1 F.c20.h558_0 F.c20.h559_0
    (Xs := ![generatedRoot.w_562_0 ctx, generatedRoot.w_576_0 ctx]) (fun j ↦ by fin_cases j; exacts [F.c21.h562_0, F.c22.h576_0])
    (D' := ![generatedRoot.w_563_0 ctx, generatedRoot.w_577_0 ctx]) (fun j ↦ by fin_cases j; exacts [F.c21.h563_0, F.c22.h577_0])
    (D := ![generatedRoot.w_564_0 ctx, generatedRoot.w_578_0 ctx]) (fun j ↦ by fin_cases j; exacts [F.c21.h564_0, F.c22.h578_0]) (payload := generatedRoot.w_568_0 ctx) hp0
    (Y := ![generatedRoot.w_566_0 ctx, generatedRoot.w_580_0 ctx])
    (A' := ![![generatedRoot.w_571_0 ctx, generatedRoot.w_584_0 ctx], ![generatedRoot.w_572_0 ctx, generatedRoot.w_585_0 ctx]])
    (fun j ↦ by fin_cases j; exacts [F.c21.h566_0, F.c22.h580_0])
    (fun j ↦ by fin_cases j; exacts [F.c21.h571_0, F.c22.h584_0])
    (fun j ↦ by fin_cases j; exacts [F.c21.h572_0, F.c22.h585_0])
  have hP1' : Rel sm ![![generatedRoot.w_571_0 ctx, generatedRoot.w_584_0 ctx], ![generatedRoot.w_572_0 ctx, generatedRoot.w_585_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (α 0 * α 2 * unitPair sm j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_570_0 ctx, generatedRoot.w_583_0 ctx] ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Ba) := by
    convert hP1 using 1
    funext j; fin_cases j <;> rfl
  have hp02 := payload_mul hp0 hp2
  have hprod := label_last (by norm_num) hq1 layout hlayout hexact hG hα02 hP1' hm3
    F.c21.h571_0_1 F.c21.h573_0 F.c22.h584_0_1 F.c22.h586_0 F.c22.h587_0
    (Xs := ![generatedRoot.w_590_0 ctx, generatedRoot.w_610_0 ctx]) (fun j ↦ by fin_cases j; exacts [F.c22.h590_0, F.c23.h610_0])
    (D' := ![generatedRoot.w_591_0 ctx, generatedRoot.w_611_0 ctx]) (fun j ↦ by fin_cases j; exacts [F.c22.h591_0, F.c23.h611_0])
    (D := ![generatedRoot.w_592_0 ctx, generatedRoot.w_612_0 ctx]) (fun j ↦ by fin_cases j; exacts [F.c23.h592_0, F.c23.h612_0]) (payload := generatedRoot.w_597_0 ctx) hp02
    (Y := ![generatedRoot.w_594_0 ctx, generatedRoot.w_614_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c23.h594_0, F.c23.h614_0])
  have hX : generatedRoot.w_602_0 ctx = generatedRoot.w_57_0 ctx :=
    gadget_rebase_one hq1 F.c23.h600_0
  have hDg : generatedRoot.w_604_0 ctx = generatedRoot.w_603_0 ctx :=
    compactRebase_self_gen (by norm_num) hq1 F.c23.h604_0
  have hdec := F.c23.h603_0
  rw [hX, ← hDg] at hdec
  have hfin := label_finish (by norm_num) hq1 layout hlayout hexact hG hα023 hprod hdec hs
    (D0 := ![generatedRoot.w_606_0 ctx, generatedRoot.w_627_0 ctx]) (D1 := ![generatedRoot.w_619_0 ctx, generatedRoot.w_629_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c23.h606_0, F.c23.h627_0])
    (fun j ↦ by fin_cases j; exacts [F.c23.h619_0, F.c23.h629_0])
  have hred : generatedRoot.w_624_0 ctx = fun _ _ ↦ reducePoly modulus_0 8 (α 0 * α 2 * α 3) := by
    rw [modulusReduce_same (by norm_num) F.c23.h624_0]
    exact payload_mul hp02 hp3
  refine Rel.c_eq hfin ?_
  funext j
  fin_cases j
  · show _ = matrixAdd (matrixAdd (matrixMul (matrixMul (generatedRoot.w_599_0 ctx)
        (generatedRoot.w_604_0 ctx)) (generatedRoot.w_606_0 ctx)) (matrixMul (matrixMul
        (generatedRoot.w_617_0 ctx) (generatedRoot.w_604_0 ctx)) (generatedRoot.w_619_0 ctx)))
        (matrixMulScalarLeft (generatedRoot.w_624_0 ctx) (generatedRoot.w_244_0 ctx))
    rw [matrixMulScalarLeft_eq, hred]
    rfl
  · show _ = matrixAdd (matrixAdd (matrixMul (matrixMul (generatedRoot.w_599_0 ctx)
        (generatedRoot.w_604_0 ctx)) (generatedRoot.w_627_0 ctx)) (matrixMul (matrixMul
        (generatedRoot.w_617_0 ctx) (generatedRoot.w_604_0 ctx)) (generatedRoot.w_629_0 ctx)))
        (matrixMulScalarLeft (generatedRoot.w_624_0 ctx) (generatedRoot.w_245_0 ctx))
    rw [matrixMulScalarLeft_eq, hred]
    rfl

set_option maxHeartbeats 64000000 in
/-- **The label target `(w_680, w_687)` of step 0**: `α₁ β` with `β` the value of the
`dio2/label-0` member `1` (`hone`, payload `w_276`). -/
theorem online0_label_680 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm : ErrorPoly 8} {α : Fin 4 → ErrorPoly 8} (hα : ∀ i, polyNorm (α i) ≤ 1) {Ba : Nat}
    (hlab : ∀ i : Fin 4, Rel sm
      (labelKeysG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 labelInPre (i : Nat))
      (fun j ↦ reducePoly modulus_0 8 (α i * unitPair sm j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_372_0 ctx i, generatedRoot.w_377_0 ctx i] Ba)
    (hpay : ∀ i : Fin 4, generatedRoot.w_391_0 ctx i = fun _ _ ↦ reducePoly modulus_0 8 (α i))
    {ts : Fin 2 → ExactMatrix modulus_0 8 1 25} {Bs : Nat}
    (hs : Rel sm ![![generatedRoot.w_402_0 ctx, generatedRoot.w_428_0 ctx],
        ![generatedRoot.w_417_0 ctx, generatedRoot.w_433_0 ctx]] ts
      ![generatedRoot.w_244_0 ctx, generatedRoot.w_245_0 ctx] Bs)
    {β : ErrorPoly 8} {Bo : Nat}
    (hone : Rel sm (labelKeysG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 labelPre
        ((1 : Fin 2) : Nat))
      (fun j ↦ reducePoly modulus_0 8 (β * unitPair sm j) • generatedRoot.w_57_0 ctx)
      ![generatedRoot.w_257_0 ctx 1, generatedRoot.w_262_0 ctx 1] Bo)
    (hpay0 : generatedRoot.w_276_0 ctx 1 = fun _ _ ↦ reducePoly modulus_0 8 β)
    (hα1β : polyNorm (α 1 * β) ≤ 1) :
    Rel sm (fun k j ↦
        rowOf (matrixMul (generatedRoot.w_642_0 ctx) (generatedRoot.w_647_0 ctx)) k.succ *
            generatedRoot.w_658_0 ctx * ![generatedRoot.w_660_0 ctx, generatedRoot.w_681_0 ctx] j +
          rowOf (matrixMul (generatedRoot.w_642_0 ctx) (generatedRoot.w_666_0 ctx)) k.succ *
            generatedRoot.w_658_0 ctx * ![generatedRoot.w_673_0 ctx, generatedRoot.w_683_0 ctx] j)
      (fun j ↦ reducePoly modulus_0 8 (α 1 * β) • ts j)
      ![generatedRoot.w_680_0 ctx, generatedRoot.w_687_0 ctx]
      (25 * 8 * (25 * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Bo) * (4096 / 2)) * (4096 / 2) +
        25 * 8 * (25 * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Bo) * (4096 / 2)) * (4096 / 2) + 8 * 1 * Bs) := by
  have hG := F.c1.h57_0
  have hp : ∀ (i : Fin 4) {x : ExactMatrix modulus_0 8 1 1},
      familyGetStatic (generatedRoot.w_391_0 ctx) (i : Nat) x →
      x = fun _ _ ↦ reducePoly modulus_0 8 (α i) := fun i _ h ↦ by
    rw [familyGetStatic_fin i h, hpay i]
  have hm1 := label_rel_of_hash 1 (hlab 1) F.c24.h635_0 F.c24.h639_0 F.c24.h636_0 F.c24.h640_0
    F.c24.h634_0 F.c24.h638_0
  have hb := label_rel_of_hash 1 hone F.c24.h643_0 F.c24.h662_0 F.c24.h644_0 F.c24.h663_0
    F.c24.h650_0 F.c25.h669_0
  have hp1 := hp 1 F.c24.h651_0
  have hpb : generatedRoot.w_676_0 ctx = fun _ _ ↦ reducePoly modulus_0 8 β := by
    rw [familyGetStatic_fin (1 : Fin 2) F.c25.h676_0, hpay0]
  have hprod := label_last (by norm_num) hq1 layout hlayout hexact hG (hα 1) hm1 hb
    F.c24.h635_0_1 F.c24.h637_0 F.c24.h639_0_1 F.c24.h641_0 F.c24.h642_0
    (Xs := ![generatedRoot.w_645_0 ctx, generatedRoot.w_664_0 ctx]) (fun j ↦ by fin_cases j; exacts [F.c24.h645_0, F.c24.h664_0])
    (D' := ![generatedRoot.w_646_0 ctx, generatedRoot.w_665_0 ctx]) (fun j ↦ by fin_cases j; exacts [F.c24.h646_0, F.c25.h665_0])
    (D := ![generatedRoot.w_647_0 ctx, generatedRoot.w_666_0 ctx]) (fun j ↦ by fin_cases j; exacts [F.c24.h647_0, F.c25.h666_0]) (payload := generatedRoot.w_651_0 ctx) hp1
    (Y := ![generatedRoot.w_649_0 ctx, generatedRoot.w_668_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c24.h649_0, F.c25.h668_0])
  have hX : generatedRoot.w_656_0 ctx = generatedRoot.w_57_0 ctx :=
    gadget_rebase_one hq1 F.c24.h654_0
  have hDg : generatedRoot.w_658_0 ctx = generatedRoot.w_657_0 ctx :=
    compactRebase_self_gen (by norm_num) hq1 F.c24.h658_0
  have hdec := F.c24.h657_0
  rw [hX, ← hDg] at hdec
  have hfin := label_finish (by norm_num) hq1 layout hlayout hexact hG hα1β hprod hdec hs
    (D0 := ![generatedRoot.w_660_0 ctx, generatedRoot.w_681_0 ctx]) (D1 := ![generatedRoot.w_673_0 ctx, generatedRoot.w_683_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c24.h660_0, F.c25.h681_0])
    (fun j ↦ by fin_cases j; exacts [F.c25.h673_0, F.c25.h683_0])
  have hred : generatedRoot.w_678_0 ctx = fun _ _ ↦ reducePoly modulus_0 8 (α 1 * β) := by
    rw [modulusReduce_same (by norm_num) F.c25.h678_0]
    exact payload_mul hp1 hpb
  refine Rel.c_eq hfin ?_
  funext j
  fin_cases j
  · show _ = matrixAdd (matrixAdd (matrixMul (matrixMul (generatedRoot.w_653_0 ctx)
        (generatedRoot.w_658_0 ctx)) (generatedRoot.w_660_0 ctx)) (matrixMul (matrixMul
        (generatedRoot.w_671_0 ctx) (generatedRoot.w_658_0 ctx)) (generatedRoot.w_673_0 ctx)))
        (matrixMulScalarLeft (generatedRoot.w_678_0 ctx) (generatedRoot.w_244_0 ctx))
    rw [matrixMulScalarLeft_eq, hred]
    rfl
  · show _ = matrixAdd (matrixAdd (matrixMul (matrixMul (generatedRoot.w_653_0 ctx)
        (generatedRoot.w_658_0 ctx)) (generatedRoot.w_681_0 ctx)) (matrixMul (matrixMul
        (generatedRoot.w_671_0 ctx) (generatedRoot.w_658_0 ctx)) (generatedRoot.w_683_0 ctx)))
        (matrixMulScalarLeft (generatedRoot.w_678_0 ctx) (generatedRoot.w_245_0 ctx))
    rw [matrixMulScalarLeft_eq, hred]
    rfl

end Dio2Proof
