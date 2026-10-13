import Part_I_Local
import TopInst
import Part_I_EncRoot
import Part_I_EncFacts
import Part_I_PreRhs
import Part_I_PreSwitch
import Part_I_OnlineChain
import Part_I_SmudgeChain
import TopHint
import TopRequirement

/-! The facts the online chains of part `I` assume about the online root's inputs (written by
`generator/gen_inputs.py`). -/

namespace RefreshTopProof.Part_I

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The inputs of the mask chain. -/
theorem maskInputs {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params' : Stage_preprocess.Params} {params'' : Stage_encrypt.Params}
    {pw_9_0 : ExactMatrix Q0 8 1 1}
    {pw_26_0 : ByteArray}
    {pw_32_0 : Fin 40 → Int}
    {pw_33_0 : Fin 40 → Int}
    {pw_37_0 : Fin 40 → Int}
    {pw_38_0 : Fin 40 → Int}
    {pw_58_0 : Int}
    {pw_60_0 : Int}
    {pw_108_0 : Int}
    {pw_110_0 : Int}
    {pw_11_0 : ExactMatrix Q0 8 1 1}
    {pw_27_0 : ExactMatrix Q7 8 1 12}
    {pw_28_0 : ExactMatrix Q7 8 1 12}
    {pw_29_0 : ExactMatrix Q7 8 1 12}
    {pw_30_0 : ExactMatrix Q7 8 1 12}
    {pw_31_0 : ExactMatrix Q7 8 1 1}
    {pw_34_0 : ExactMatrix Q7 8 1 1}
    {pw_35_0 : Fin 40 → ExactMatrix Q6 8 1 10}
    {pw_35_1 : Fin 40 → ExactMatrix Q6 8 1 10}
    {pw_35_2 : Fin 40 → ExactMatrix Q6 8 1 10}
    {pw_35_3 : Fin 40 → ExactMatrix Q6 8 1 10}
    {pw_35_4 : Fin 40 → ExactMatrix Q6 8 1 10}
    {pw_35_5 : Fin 40 → ExactMatrix Q6 8 1 10}
    {pw_35_6 : Fin 40 → ExactMatrix Q6 8 1 10}
    {pw_35_7 : Fin 40 → ExactMatrix Q6 8 1 10}
    {pw_36_0 : Fin 8 → ExactMatrix Q6 8 1 10}
    {pw_36_1 : Fin 8 → ExactMatrix Q6 8 1 10}
    {pw_36_2 : Fin 8 → ExactMatrix Q6 8 1 10}
    {pw_36_3 : Fin 8 → ExactMatrix Q6 8 1 10}
    {pw_39_0 : Fin 40 → ExactMatrix Q5 8 1 8}
    {pw_39_1 : Fin 40 → ExactMatrix Q5 8 1 8}
    {pw_39_2 : Fin 40 → ExactMatrix Q5 8 1 8}
    {pw_39_3 : Fin 40 → ExactMatrix Q5 8 1 8}
    {pw_39_4 : Fin 40 → ExactMatrix Q5 8 1 8}
    {pw_39_5 : Fin 40 → ExactMatrix Q5 8 1 8}
    {pw_39_6 : Fin 40 → ExactMatrix Q5 8 1 8}
    {pw_39_7 : Fin 40 → ExactMatrix Q5 8 1 8}
    {pw_41_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {pw_41_1 : Fin 4 → ExactMatrix Q0 8 1 6}
    {pw_41_2 : Fin 4 → ExactMatrix Q0 8 1 6}
    {pw_41_3 : Fin 4 → ExactMatrix Q0 8 1 6}
    {pw_43_0 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_43_1 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_43_2 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_43_3 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_47_0 : ExactMatrix Q6 8 1 10}
    {pw_52_0 : ExactMatrix Q6 8 1 1}
    {pw_53_0 : ExactMatrix Q5 8 1 10}
    {pw_54_0 : ExactMatrix Q6 8 1 10}
    {pw_55_0 : ExactMatrix Q6 8 1 10}
    {pw_56_0 : ExactMatrix Q6 8 1 1}
    {pw_57_0 : ExactMatrix Q6 8 1 10}
    {pw_59_0 : ExactMatrix Q7 8 1 1}
    {pw_61_0 : ExactMatrix Q7 8 1 1}
    {pw_62_0 : ExactMatrix Q6 8 1 1}
    {pw_63_0 : ExactMatrix Q6 8 1 10}
    {pw_64_0 : ExactMatrix Q6 8 1 10}
    {pw_65_0 : ExactMatrix Q6 8 1 10}
    {pw_66_0 : ExactMatrix Q6 8 1 10}
    {pw_67_0 : ExactMatrix Q6 8 1 10}
    {pw_77_0 : ExactMatrix Q5 8 1 8}
    {pw_81_0 : ExactMatrix Q6 8 1 10}
    {pw_85_0 : ExactMatrix Q6 8 1 1}
    {pw_86_0 : ExactMatrix Q6 8 1 10}
    {pw_87_0 : ExactMatrix Q6 8 1 10}
    {pw_88_0 : ExactMatrix Q6 8 1 10}
    {pw_89_0 : ExactMatrix Q6 8 1 10}
    {pw_90_0 : ExactMatrix Q6 8 1 10}
    {pw_91_0 : ExactMatrix Q6 8 1 10}
    {pw_92_0 : ExactMatrix Q6 8 1 10}
    {pw_102_0 : ExactMatrix Q5 8 1 1}
    {pw_103_0 : ExactMatrix Q0 8 1 8}
    {pw_104_0 : ExactMatrix Q5 8 1 8}
    {pw_105_0 : ExactMatrix Q5 8 1 8}
    {pw_106_0 : ExactMatrix Q5 8 1 1}
    {pw_107_0 : ExactMatrix Q5 8 1 8}
    {pw_109_0 : ExactMatrix Q7 8 1 1}
    {pw_111_0 : ExactMatrix Q7 8 1 1}
    {pw_112_0 : ExactMatrix Q5 8 1 1}
    {pw_113_0 : ExactMatrix Q5 8 1 8}
    {pw_114_0 : ExactMatrix Q5 8 1 8}
    {pw_115_0 : ExactMatrix Q5 8 1 8}
    {pw_116_0 : ExactMatrix Q5 8 1 8}
    {pw_117_0 : ExactMatrix Q5 8 1 8}
    {pw_126_0 : ExactMatrix Q5 8 1 8}
    {pw_130_0 : ExactMatrix Q5 8 1 1}
    {pw_131_0 : ExactMatrix Q5 8 1 8}
    {pw_132_0 : ExactMatrix Q5 8 1 8}
    {pw_133_0 : ExactMatrix Q5 8 1 8}
    {pw_134_0 : ExactMatrix Q5 8 1 8}
    {pw_135_0 : ExactMatrix Q5 8 1 8}
    {pw_136_0 : ExactMatrix Q5 8 1 8}
    {pw_137_0 : ExactMatrix Q5 8 1 8}
    (pe_11_0 : pw_11_0 = (1 : ExactMatrix Q0 8 1 1))
    (pe_55_0 : pw_55_0 = matrixMulScalarLeft pw_52_0 pw_54_0)
    (pe_56_0 : pw_56_0 = (matrixPolynomial [(p5)] : ExactMatrix Q6 8 1 1))
    (pe_57_0 : pw_57_0 = matrixMulScalarRight pw_55_0 pw_56_0)
    (pe_59_0 : pw_59_0 = liftInteger pw_58_0)
    (pe_61_0 : pw_61_0 = multiplyMonomial pw_59_0 pw_60_0)
    (pe_63_0 : pw_63_0 = matrixMulScalarRight pw_57_0 pw_62_0)
    (pe_64_0 : pw_64_0 = matrixMulScalarLeft pw_52_0 pw_47_0)
    (pe_65_0 : pw_65_0 = matrixSub pw_63_0 pw_64_0)
    (pe_67_0 : pw_67_0 = matrixAdd pw_65_0 pw_66_0)
    (pe_86_0 : pw_86_0 = matrixMulScalarLeft pw_85_0 pw_54_0)
    (pe_87_0 : pw_87_0 = matrixMulScalarRight pw_86_0 pw_56_0)
    (pe_88_0 : pw_88_0 = matrixMulScalarRight pw_87_0 pw_62_0)
    (pe_89_0 : pw_89_0 = matrixMulScalarLeft pw_52_0 pw_81_0)
    (pe_90_0 : pw_90_0 = matrixSub pw_88_0 pw_89_0)
    (pe_92_0 : pw_92_0 = matrixAdd pw_90_0 pw_91_0)
    (pe_105_0 : pw_105_0 = matrixMulScalarLeft pw_102_0 pw_104_0)
    (pe_106_0 : pw_106_0 = (matrixPolynomial [(p4)] : ExactMatrix Q5 8 1 1))
    (pe_107_0 : pw_107_0 = matrixMulScalarRight pw_105_0 pw_106_0)
    (pe_109_0 : pw_109_0 = liftInteger pw_108_0)
    (pe_111_0 : pw_111_0 = multiplyMonomial pw_109_0 pw_110_0)
    (pe_113_0 : pw_113_0 = matrixMulScalarRight pw_107_0 pw_112_0)
    (pe_114_0 : pw_114_0 = matrixMulScalarLeft pw_102_0 pw_77_0)
    (pe_115_0 : pw_115_0 = matrixSub pw_113_0 pw_114_0)
    (pe_117_0 : pw_117_0 = matrixAdd pw_115_0 pw_116_0)
    (pe_131_0 : pw_131_0 = matrixMulScalarLeft pw_130_0 pw_104_0)
    (pe_132_0 : pw_132_0 = matrixMulScalarRight pw_131_0 pw_106_0)
    (pe_133_0 : pw_133_0 = matrixMulScalarRight pw_132_0 pw_112_0)
    (pe_134_0 : pw_134_0 = matrixMulScalarLeft pw_102_0 pw_126_0)
    (pe_135_0 : pw_135_0 = matrixSub pw_133_0 pw_134_0)
    (pe_137_0 : pw_137_0 = matrixAdd pw_135_0 pw_136_0)
    (ph31_0 : centeredRebaseRuns pw_9_0 pw_31_0)
    (ph34_0 : centeredRebaseRuns pw_11_0 pw_34_0)
    (ph47_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal ((Stage_preprocess.sel_4 ((Int.ofNat i0)))), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal ((Stage_preprocess.sel_5 ((Int.ofNat i0)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (pw_26_0) pw_47_0)
    (ph52_0 : modulusReduceRuns pw_31_0 pw_52_0)
    (ph53_0 : gadgetMatrixRuns Backend.backend 32768 10 pw_53_0)
    (ph54_0 : centeredRebaseRuns pw_53_0 pw_54_0)
    (ph62_0 : modulusReduceRuns pw_61_0 pw_62_0)
    (ph66_0 : gaussianSample ((4/1)) (26) pw_66_0)
    (ph81_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal ((Stage_preprocess.sel_4 ((Int.ofNat i0)))), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal ((Stage_preprocess.sel_5 ((Int.ofNat i0)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (pw_26_0) pw_81_0)
    (ph85_0 : modulusReduceRuns pw_34_0 pw_85_0)
    (ph91_0 : gaussianSample ((4/1)) (26) pw_91_0)
    (ph102_0 : modulusReduceRuns pw_31_0 pw_102_0)
    (ph103_0 : gadgetMatrixRuns Backend.backend 32768 8 pw_103_0)
    (ph104_0 : centeredRebaseRuns pw_103_0 pw_104_0)
    (ph112_0 : modulusReduceRuns pw_111_0 pw_112_0)
    (ph77_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 51, 41, 47, 49, 47, 50]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal ((Stage_preprocess.sel_4 ((Int.ofNat i0)))), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal ((Stage_preprocess.sel_5 ((Int.ofNat i0)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (pw_26_0) pw_77_0)
    (ph126_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 51, 41, 47, 49, 47, 50]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal ((Stage_preprocess.sel_4 ((Int.ofNat i0)))), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal ((Stage_preprocess.sel_5 ((Int.ofNat i0)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (pw_26_0) pw_126_0)
    (ph116_0 : gaussianSample ((4/1)) (26) pw_116_0)
    (ph130_0 : modulusReduceRuns pw_34_0 pw_130_0)
    (ph136_0 : gaussianSample ((4/1)) (26) pw_136_0)
    (ph35_7 : (∀ i : Fin 40, Stage_preprocess.parallel_parallel_generatedRoot_407_35 Backend.backend model params' i0 i (pw_26_0, (pw_27_0, (pw_28_0, (pw_29_0, (pw_30_0, (pw_31_0, ((pw_32_0 i), ((pw_33_0 i), (pw_34_0, ()))))))))) ((pw_35_0 i), ((pw_35_1 i), ((pw_35_2 i), ((pw_35_3 i), ((pw_35_4 i), ((pw_35_5 i), ((pw_35_6 i), ((pw_35_7 i), ()))))))))))
    (ph39_7 : (∀ i : Fin 40, Stage_preprocess.parallel_parallel_generatedRoot_407_39 Backend.backend model params' i0 i (pw_26_0, (pw_36_0, (pw_36_2, (pw_36_1, (pw_36_3, (pw_31_0, ((pw_37_0 i), ((pw_38_0 i), (pw_34_0, ()))))))))) ((pw_39_0 i), ((pw_39_1 i), ((pw_39_2 i), ((pw_39_3 i), ((pw_39_4 i), ((pw_39_5 i), ((pw_39_6 i), ((pw_39_7 i), ()))))))))))
    (ph41_3 : (∀ i : Fin 4, Stage_preprocess.parallel_parallel_generatedRoot_407_41 Backend.backend model params' i0 i (pw_26_0, (pw_31_0, (pw_34_0, ()))) ((pw_41_0 i), ((pw_41_1 i), ((pw_41_2 i), ((pw_41_3 i), ()))))))
    (ph43_3 : (∀ i : Fin 4, Stage_preprocess.parallel_parallel_generatedRoot_407_43 Backend.backend model params' i0 i (pw_26_0, (pw_31_0, (pw_34_0, ()))) ((pw_43_0 i), ((pw_43_1 i), ((pw_43_2 i), ((pw_43_3 i), ()))))))
    {qw_4_0 : ExactMatrix Q0 8 1 1}
    {qw_11_0 : ExactMatrix Q0 8 1 1}
    {qw_337_0 : ExactMatrix Q7 8 1 12}
    {qw_338_0 : ExactMatrix Q7 8 1 12}
    {qw_339_0 : ExactMatrix Q7 8 1 12}
    {qw_340_0 : ExactMatrix Q7 8 1 12}
    {qw_351_0 : Int}
    {qw_353_0 : Int}
    {qw_369_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {qw_370_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {qw_371_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {qw_372_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {qw_373_0 : Fin 5 → Int}
    {qw_374_0 : Fin 5 → Int}
    {qw_376_0 : ByteArray}
    {qw_7_0 : ExactMatrix Q0 8 1 1}
    {qw_29_0 : ExactMatrix Q0 8 1 1}
    {qw_341_0 : ExactMatrix Q7 8 1 1}
    {qw_342_0 : ExactMatrix Q7 8 1 12}
    {qw_343_0 : ExactMatrix Q7 8 1 1}
    {qw_344_0 : ExactMatrix Q7 8 1 12}
    {qw_345_0 : ExactMatrix Q7 8 1 12}
    {qw_346_0 : ExactMatrix Q7 8 1 1}
    {qw_347_0 : ExactMatrix Q6 8 1 12}
    {qw_348_0 : ExactMatrix Q7 8 1 12}
    {qw_349_0 : ExactMatrix Q7 8 1 12}
    {qw_350_0 : ExactMatrix Q7 8 1 1}
    {qw_352_0 : ExactMatrix Q7 8 1 1}
    {qw_354_0 : ExactMatrix Q7 8 1 1}
    {qw_355_0 : ExactMatrix Q7 8 1 1}
    {qw_356_0 : ExactMatrix Q7 8 1 12}
    {qw_357_0 : ExactMatrix Q7 8 1 12}
    {qw_358_0 : ExactMatrix Q7 8 1 12}
    {qw_359_0 : ExactMatrix Q7 8 1 12}
    {qw_360_0 : ExactMatrix Q7 8 1 12}
    {qw_361_0 : ExactMatrix Q7 8 1 12}
    {qw_362_0 : ExactMatrix Q7 8 1 12}
    {qw_363_0 : ExactMatrix Q7 8 1 1}
    {qw_364_0 : ExactMatrix Q7 8 1 12}
    {qw_365_0 : ExactMatrix Q7 8 1 12}
    {qw_366_0 : ExactMatrix Q7 8 1 12}
    {qw_367_0 : ExactMatrix Q7 8 1 12}
    {qw_368_0 : ExactMatrix Q7 8 1 12}
    {qw_375_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {qw_375_1 : Fin 5 → ExactMatrix Q7 8 1 14}
    {qw_375_2 : Fin 5 → ExactMatrix Q7 8 1 14}
    {qw_375_3 : Fin 5 → ExactMatrix Q7 8 1 14}
    {qw_375_4 : Fin 5 → ExactMatrix Q7 8 1 14}
    {qw_375_5 : Fin 5 → ExactMatrix Q7 8 1 14}
    {qw_375_6 : Fin 5 → ExactMatrix Q7 8 1 1}
    {qw_379_0 : Fin 4 → ExactMatrix Q5 8 1 8}
    {qw_379_1 : Fin 4 → ExactMatrix Q5 8 1 8}
    {qw_380_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {qw_380_1 : Fin 4 → ExactMatrix Q0 8 1 6}
    (qe_7_0 : qw_7_0 = (1 : ExactMatrix Q0 8 1 1))
    (qe_29_0 : qw_29_0 = (1 : ExactMatrix Q0 8 1 1))
    (qe_342_0 : qw_342_0 = matrixMulScalarLeft qw_341_0 qw_337_0)
    (qe_344_0 : qw_344_0 = matrixMulScalarLeft qw_343_0 qw_339_0)
    (qe_345_0 : qw_345_0 = matrixAdd qw_342_0 qw_344_0)
    (qe_349_0 : qw_349_0 = matrixMulScalarLeft qw_346_0 qw_348_0)
    (qe_350_0 : qw_350_0 = (matrixPolynomial [(p6)] : ExactMatrix Q7 8 1 1))
    (qe_352_0 : qw_352_0 = liftInteger qw_351_0)
    (qe_354_0 : qw_354_0 = multiplyMonomial qw_352_0 qw_353_0)
    (qe_355_0 : qw_355_0 = matrixMulScalarLeft qw_350_0 qw_354_0)
    (qe_356_0 : qw_356_0 = matrixMulScalarRight qw_349_0 qw_355_0)
    (qe_357_0 : qw_357_0 = matrixSub qw_345_0 qw_356_0)
    (qe_359_0 : qw_359_0 = matrixAdd qw_357_0 qw_358_0)
    (qe_360_0 : qw_360_0 = matrixMulScalarLeft qw_341_0 qw_338_0)
    (qe_361_0 : qw_361_0 = matrixMulScalarLeft qw_343_0 qw_340_0)
    (qe_362_0 : qw_362_0 = matrixAdd qw_360_0 qw_361_0)
    (qe_364_0 : qw_364_0 = matrixMulScalarLeft qw_363_0 qw_348_0)
    (qe_365_0 : qw_365_0 = matrixMulScalarRight qw_364_0 qw_355_0)
    (qe_366_0 : qw_366_0 = matrixSub qw_362_0 qw_365_0)
    (qe_368_0 : qw_368_0 = matrixAdd qw_366_0 qw_367_0)
    (qh375_6 : (∀ i : Fin 5, Stage_encrypt.parallel_parallel_generatedRoot_454_375 Backend.backend model params'' i0 i ((qw_369_0 i), ((qw_370_0 i), ((qw_371_0 i), ((qw_372_0 i), (qw_341_0, (qw_343_0, ((qw_373_0 i), ((qw_374_0 i), ())))))))) ((qw_375_0 i), ((qw_375_1 i), ((qw_375_2 i), ((qw_375_3 i), ((qw_375_4 i), ((qw_375_5 i), ((qw_375_6 i), ())))))))))
    (qh379_1 : (∀ i : Fin 4, Stage_encrypt.parallel_parallel_generatedRoot_454_379 Backend.backend model params'' i0 i (qw_341_0, (qw_376_0, (qw_343_0, ()))) ((qw_379_0 i), ((qw_379_1 i), ()))))
    (qh380_1 : (∀ i : Fin 4, Stage_encrypt.parallel_parallel_generatedRoot_454_380 Backend.backend model params'' i0 i (qw_341_0, (qw_376_0, (qw_343_0, ()))) ((qw_380_0 i), ((qw_380_1 i), ()))))
    (qh341_0 : centeredRebaseRuns qw_4_0 qw_341_0)
    (qh343_0 : centeredRebaseRuns qw_7_0 qw_343_0)
    (qh346_0 : centeredRebaseRuns qw_11_0 qw_346_0)
    (qh347_0 : gadgetMatrixRuns Backend.backend 32768 12 qw_347_0)
    (qh348_0 : centeredRebaseRuns qw_347_0 qw_348_0)
    (qh358_0 : gaussianSample ((4/1)) (26) qw_358_0)
    (qh363_0 : centeredRebaseRuns qw_29_0 qw_363_0)
    (qh367_0 : gaussianSample ((4/1)) (26) qw_367_0)
    {r sm : ErrorPoly N} (hr : polyNorm r ≤ 1) (hsm : polyNorm sm ≤ 1)
    (hpay : pw_9_0 = fun _ _ ↦ reducePoly Q0 N r) (qhm : qw_4_0 = fun _ _ ↦ reducePoly Q0 N sm)
    (qhp : qw_11_0 = fun _ _ ↦ reducePoly Q0 N r) (q351 : qw_351_0 = 1) (q353 : qw_353_0 = 0)
    (p58 : pw_58_0 = 1) (p60 : pw_60_0 = 0) (p108 : pw_108_0 = 1) (p110 : pw_110_0 = 0)
    {GQ7_14 : ExactMatrix Q7 N 1 14} {GQ6_12 : ExactMatrix Q6 N 1 12} {RQ7_12 : ExactMatrix Q7 N 1 12}
    {GQ5_10 : ExactMatrix Q5 N 1 10} {RQ6_10 : ExactMatrix Q6 N 1 10}
    {GQ0_8 : ExactMatrix Q0 N 1 8} {RQ5_8 : ExactMatrix Q5 N 1 8}
    {GQ1_6 : ExactMatrix (R i0) N 1 6} {RQ0_6 : ExactMatrix Q0 N 1 6}
    (hGQ7_14 : gadgetMatrixRuns Backend.backend 32768 14 GQ7_14)
    (hGQ6_12 : gadgetMatrixRuns Backend.backend 32768 12 GQ6_12) (hRQ7_12 : centeredRebaseRuns GQ6_12 RQ7_12)
    (hGQ5_10 : gadgetMatrixRuns Backend.backend 32768 10 GQ5_10) (hRQ6_10 : centeredRebaseRuns GQ5_10 RQ6_10)
    (hGQ0_8 : gadgetMatrixRuns Backend.backend 32768 8 GQ0_8) (hRQ5_8 : centeredRebaseRuns GQ0_8 RQ5_8)
    (hGQ1_6 : gadgetMatrixRuns Backend.backend 32768 6 GQ1_6) (hRQ0_6 : centeredRebaseRuns GQ1_6 RQ0_6)
    (hlc : ∀ k, qw_373_0 k * qw_373_0 k = 1)
    (hlr : ∀ k, qw_374_0 k * 2 + (1 - qw_373_0 k) * 8 = (labelExp k : Int) * 4)
    (hle : ∀ k, 0 ≤ qw_374_0 k ∧ qw_374_0 k ≤ 7)
    (hs01 : ∀ i, 0 ≤ pw_32_0 i ∧ pw_32_0 i ≤ 1) (hs01' : ∀ i, 0 ≤ pw_37_0 i ∧ pw_37_0 i ≤ 1)
    (he0 : ∀ i, pw_33_0 i = 0) (he0' : ∀ i, pw_38_0 i = 0)
    (hsum : ∀ o : Fin 8, ∑ k : Fin 5, pw_32_0 (cellIndex o k) = 1)
    (hsum' : ∀ o : Fin 8, ∑ k : Fin 5, pw_37_0 (cellIndex o k) = 1) :
    Rel sm ![![qw_337_0, qw_338_0], ![qw_339_0, qw_340_0]]
      (fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) * 1 * unitPair r j) • RQ7_12) ![qw_359_0, qw_368_0] 26 ∧
    (∃ eB : Fin 2 → ErrorMatrix N 1 10, (∀ j, CoeffBound (eB j) 26) ∧ ∀ j,
      reducePoly Q6 N r • ![pw_47_0, pw_81_0] j + ![pw_67_0, pw_92_0] j =
        reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair r j) • RQ6_10 + reduceMatrix Q6 N 1 10 (eB j)) ∧
    (∃ eB : Fin 2 → ErrorMatrix N 1 8, (∀ j, CoeffBound (eB j) 26) ∧ ∀ j,
      reducePoly Q5 N r • ![pw_77_0, pw_126_0] j + ![pw_117_0, pw_137_0] j =
        reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * 0) * unitPair r j) • RQ5_8 +
          reduceMatrix Q5 N 1 8 (eB j)) ∧
    (∀ i : Fin 4, Rel sm (lutA model qw_376_0 Q5 8 2 (sigma i))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • RQ5_8) ![qw_379_0 i, qw_379_1 i] 26) ∧
    (∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q0 N (substitution (sigma i).toNat (sigma_odd i) r) •
          ![![pw_41_0, pw_41_1], ![pw_41_2, pw_41_3]] 0 j i + ![![pw_41_0, pw_41_1], ![pw_41_2, pw_41_3]] 1 j i =
        (lutCoefficient Q0 996917851236477127152305493547100497 (fun k ↦ reducePoly Q0 N (readoutTable k))
          (sigma i).toNat * reducePoly Q0 N (((((qd i0) : Nat) : Int) : ErrorPoly N) * unitPair r j)) • RQ0_6 +
            reduceMatrix Q0 N 1 6 (e j)) ∧
    (∀ i : Fin 4, Rel sm (lutA model qw_376_0 Q0 6 3 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • RQ0_6) ![qw_380_0 i, qw_380_1 i] 26) ∧
    (∀ k : Fin 5, Rel sm ![![qw_375_0 k, qw_375_1 k], ![qw_375_2 k, qw_375_3 k]]
      (fun j ↦ reducePoly Q7 N (Xe (n := N) (2 * labelExp k) * unitPair sm j) • GQ7_14)
      ![qw_375_4 k, qw_375_5 k] 26) ∧
    (∃ eBm : Fin 40 → Fin 2 → ErrorMatrix N 1 10, (∀ i j, CoeffBound (eBm i j) 26) ∧ ∀ (i : Fin 40) j,
      reducePoly Q6 N r • rhsTop model pw_26_0 Q6 10 (maskRhs0 i0) i j + ![pw_35_6 i, pw_35_7 i] j =
        reducePoly Q6 N (((p5 : Nat) : Int) * ((pw_32_0 i : Int) : ErrorPoly N) * unitPair r j) • RQ6_10 +
          reduceMatrix Q6 N 1 10 (eBm i j)) ∧
    (∃ eBm : Fin 40 → Fin 2 → ErrorMatrix N 1 8, (∀ i j, CoeffBound (eBm i j) 26) ∧ ∀ (i : Fin 40) j,
      reducePoly Q5 N r • rhsTop model pw_26_0 Q5 8 (maskRhs1 i0) i j + ![pw_39_6 i, pw_39_7 i] j =
        reducePoly Q5 N (((p4 : Nat) : Int) * ((pw_37_0 i : Int) : ErrorPoly N) * unitPair r j) • RQ5_8 +
          reduceMatrix Q5 N 1 8 (eBm i j)) ∧
    (∃ sel : Fin 8 → Fin 5, ∀ o k, pw_32_0 (cellIndex o k) = if k = sel o then 1 else 0) ∧
    (∃ sel : Fin 8 → Fin 5, ∀ o k, pw_37_0 (cellIndex o k) = if k = sel o then 1 else 0) ∧
    (∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) r) •
          ![![pw_43_0, pw_43_1], ![pw_43_2, pw_43_3]] 0 j i + ![![pw_43_0, pw_43_1], ![pw_43_2, pw_43_3]] 1 j i =
        (lutCoefficient (R i0) (inv4 i0) (fun k ↦ reducePoly (R i0) N
          (![((0 : Int) : ErrorPoly N), (((hmZ (i0 : Int)) : Int) : ErrorPoly N), ((0 : Int) : ErrorPoly N),
            (((hmZ (i0 : Int)) : Int) : ErrorPoly N)] k)) (sigma i).toNat *
          reducePoly (R i0) N (((1 : Int) : ErrorPoly N) * unitPair r j)) • GQ1_6 + reduceMatrix (R i0) N 1 6 (e j)) := by
  have hone : qw_29_0 = 1 := qe_29_0
  have h1 := encOne_359_0 (model := model) (e_7_0 := qe_7_0) (e_29_0 := qe_29_0) (e_342_0 := qe_342_0) (e_344_0 := qe_344_0) (e_345_0 := qe_345_0) (e_349_0 := qe_349_0) (e_350_0 := qe_350_0) (e_352_0 := qe_352_0) (e_354_0 := qe_354_0) (e_355_0 := qe_355_0) (e_356_0 := qe_356_0) (e_357_0 := qe_357_0) (e_359_0 := qe_359_0) (e_360_0 := qe_360_0) (e_361_0 := qe_361_0) (e_362_0 := qe_362_0) (e_364_0 := qe_364_0) (e_365_0 := qe_365_0) (e_366_0 := qe_366_0) (e_368_0 := qe_368_0) (h341_0 := qh341_0) (h343_0 := qh343_0) (h346_0 := qh346_0) (h347_0 := qh347_0) (h348_0 := qh348_0) (h358_0 := qh358_0) (h363_0 := qh363_0) (h367_0 := qh367_0) (hsm := hsm) (hsp := hr)
    (hsmIn := qhm) (hspIn := qhp) (hone := hone) (hg' := hGQ6_12) (hg := hRQ7_12)
  rw [q351, q353, message_one_zero] at h1
  have hsec : pw_31_0 = fun _ _ ↦ reducePoly Q7 N r :=
    rebase_small (q := Q0) (by inst_num) hr (by rw [← hpay]; exact ph31_0)
  have h34 : pw_34_0 = 1 := rebase_one (q := Q0) (by inst_num) (by rw [← pe_11_0]; exact ph34_0)
  rw [gadgetMatrixRuns_unique ph53_0 hGQ5_10] at ph54_0
  have e54 := centeredRebaseRuns_unique ph54_0 hRQ6_10
  rw [gadgetMatrixRuns_unique ph103_0 hGQ0_8] at ph104_0
  have e104 := centeredRebaseRuns_unique ph104_0 hRQ5_8
  subst e54 e104
  have hsmQ7 : qw_341_0 = fun _ _ ↦ reducePoly Q7 N sm :=
    rebase_small (q := Q0) (by inst_num) hsm (by rw [← qhm]; exact qh341_0)
  have h343 : qw_343_0 = 1 := rebase_one (q := Q0) (by inst_num) (by rw [← qe_7_0]; exact qh343_0)
  have hread : (fun k ↦ reducePoly Q0 N (readoutTable k)) =
      fun k ↦ reducePoly Q0 N (![Xe (n := N) 2, Xe (n := N) 2, Xe (n := N) 6, Xe (n := N) 6] k) := by
    funext k; fin_cases k <;> rfl
  refine ⟨h1, ?_, ?_, fun i ↦ encHelper379 i (qh379_1 i) hsmQ7 h343 hGQ0_8 hRQ5_8, fun i ↦ ?_,
    fun i ↦ encHelper380 i (qh380_1 i) hsmQ7 h343 hGQ1_6 hRQ0_6,
    encLabelRel375 qh375_6 hsmQ7 h343 hGQ7_14 hlc hlr hle, ?_, ?_, ?_, ?_,
    fun i ↦ ?_⟩
  · have := offset_rhs (q := Q6) (qs := Q7) (p := p5) (by inst_num) (by inst_num) hsec ph52_0
      (by rw [← h34]; exact ph85_0) (c := pw_58_0) (e := pw_60_0) (msg := pw_62_0)
      (by rw [← pe_59_0, ← pe_61_0]; exact ph62_0) ph66_0 ph91_0
      (top0 := pw_47_0) (top1 := pw_81_0) (G := pw_54_0)
    rw [p58, p60, message_one_zero] at this
    simp only [pe_67_0, pe_65_0, pe_63_0, pe_57_0, pe_55_0, pe_56_0, pe_64_0, pe_92_0, pe_90_0, pe_88_0,
      pe_87_0, pe_86_0, pe_89_0]
    exact this
  · have := offset_rhs (q := Q5) (qs := Q7) (p := p4) (by inst_num) (by inst_num) hsec ph102_0
      (by rw [← h34]; exact ph130_0) (c := pw_108_0) (e := pw_110_0) (msg := pw_112_0)
      (by rw [← pe_109_0, ← pe_111_0]; exact ph112_0) ph116_0 ph136_0
      (top0 := pw_77_0) (top1 := pw_126_0) (G := pw_104_0)
    rw [p108, p110, message_one_zero] at this
    rw [show Xe (n := N) (2 * 0) = 1 by simp [Xe]]
    simp only [pe_117_0, pe_115_0, pe_113_0, pe_107_0, pe_105_0, pe_106_0, pe_114_0, pe_137_0, pe_135_0,
      pe_133_0, pe_132_0, pe_131_0, pe_134_0]
    exact this
  · obtain ⟨e, he, hs⟩ := preSwitch41 i (ph41_3 i) hsec h34 hGQ1_6 hRQ0_6
    refine ⟨e, he, fun j ↦ ?_⟩
    rw [hread]
    fin_cases j
    · cast_exact (hs 0)
    · cast_exact (hs 1)
  · have hc := fun i : Fin 40 ↦ preRhs35 (ph35_7 i) hsec h34 hGQ5_10 hRQ6_10
    choose eB heB hrel using hc
    refine ⟨eB, heB, fun i j ↦ ?_⟩
    have ht := preTop35 (ph35_7 i)
    have := hrel i j
    rw [ht, he0 i, message_zero] at this
    exact this
  · have hc := fun i : Fin 40 ↦ preRhs39 (ph39_7 i) hsec h34 hGQ0_8 hRQ5_8
    choose eB heB hrel using hc
    refine ⟨eB, heB, fun i j ↦ ?_⟩
    have ht := preTop39 (ph39_7 i)
    have := hrel i j
    rw [ht, he0' i, message_zero] at this
    exact this
  · choose sel hsel using fun o : Fin 8 ↦ onehot_of_sum (fun k ↦ hs01 (cellIndex o k)) (hsum o)
    exact ⟨sel, hsel⟩
  · choose sel hsel using fun o : Fin 8 ↦ onehot_of_sum (fun k ↦ hs01' (cellIndex o k)) (hsum' o)
    exact ⟨sel, hsel⟩
  · obtain ⟨e, he, hs⟩ := preSwitch43 i (ph43_3 i) hsec h34 hGQ1_6
    refine ⟨e, he, fun j ↦ ?_⟩
    fin_cases j
    · cast_exact (hs 0)
    · cast_exact (hs 1)

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The inputs of the fresh chain. -/
theorem freshInputs {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params' : Stage_preprocess.Params} {params'' : Stage_encrypt.Params}
    {pw_9_0 : ExactMatrix Q0 8 1 1}
    {pw_201_0 : ByteArray}
    {pw_207_0 : Fin 240 → Int}
    {pw_208_0 : Fin 240 → Int}
    {pw_212_0 : Fin 240 → Int}
    {pw_213_0 : Fin 240 → Int}
    {pw_11_0 : ExactMatrix Q0 8 1 1}
    {pw_202_0 : ExactMatrix Q7 8 1 12}
    {pw_203_0 : ExactMatrix Q7 8 1 12}
    {pw_204_0 : ExactMatrix Q7 8 1 12}
    {pw_205_0 : ExactMatrix Q7 8 1 12}
    {pw_206_0 : ExactMatrix Q7 8 1 1}
    {pw_209_0 : ExactMatrix Q7 8 1 1}
    {pw_210_0 : Fin 240 → ExactMatrix Q6 8 1 10}
    {pw_210_1 : Fin 240 → ExactMatrix Q6 8 1 10}
    {pw_210_2 : Fin 240 → ExactMatrix Q6 8 1 10}
    {pw_210_3 : Fin 240 → ExactMatrix Q6 8 1 10}
    {pw_210_4 : Fin 240 → ExactMatrix Q6 8 1 10}
    {pw_210_5 : Fin 240 → ExactMatrix Q6 8 1 10}
    {pw_210_6 : Fin 240 → ExactMatrix Q6 8 1 10}
    {pw_210_7 : Fin 240 → ExactMatrix Q6 8 1 10}
    {pw_211_0 : Fin 48 → ExactMatrix Q6 8 1 10}
    {pw_211_1 : Fin 48 → ExactMatrix Q6 8 1 10}
    {pw_211_2 : Fin 48 → ExactMatrix Q6 8 1 10}
    {pw_211_3 : Fin 48 → ExactMatrix Q6 8 1 10}
    {pw_214_0 : Fin 240 → ExactMatrix Q5 8 1 8}
    {pw_214_1 : Fin 240 → ExactMatrix Q5 8 1 8}
    {pw_214_2 : Fin 240 → ExactMatrix Q5 8 1 8}
    {pw_214_3 : Fin 240 → ExactMatrix Q5 8 1 8}
    {pw_214_4 : Fin 240 → ExactMatrix Q5 8 1 8}
    {pw_214_5 : Fin 240 → ExactMatrix Q5 8 1 8}
    {pw_214_6 : Fin 240 → ExactMatrix Q5 8 1 8}
    {pw_214_7 : Fin 240 → ExactMatrix Q5 8 1 8}
    {pw_216_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {pw_216_1 : Fin 4 → ExactMatrix Q0 8 1 6}
    {pw_216_2 : Fin 4 → ExactMatrix Q0 8 1 6}
    {pw_216_3 : Fin 4 → ExactMatrix Q0 8 1 6}
    {pw_218_0 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_218_1 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_218_2 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_218_3 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    (pe_11_0 : pw_11_0 = (1 : ExactMatrix Q0 8 1 1))
    (ph206_0 : centeredRebaseRuns pw_9_0 pw_206_0)
    (ph209_0 : centeredRebaseRuns pw_11_0 pw_209_0)
    (ph210_7 : (∀ i : Fin 240, Stage_preprocess.parallel_parallel_generatedRoot_407_212 Backend.backend model params' i0 i (pw_201_0, (pw_202_0, (pw_203_0, (pw_204_0, (pw_205_0, (pw_206_0, ((pw_207_0 i), ((pw_208_0 i), (pw_209_0, ()))))))))) ((pw_210_0 i), ((pw_210_1 i), ((pw_210_2 i), ((pw_210_3 i), ((pw_210_4 i), ((pw_210_5 i), ((pw_210_6 i), ((pw_210_7 i), ()))))))))))
    (ph214_7 : (∀ i : Fin 240, Stage_preprocess.parallel_parallel_generatedRoot_407_216 Backend.backend model params' i0 i (pw_201_0, (pw_211_0, (pw_211_2, (pw_211_1, (pw_211_3, (pw_206_0, ((pw_212_0 i), ((pw_213_0 i), (pw_209_0, ()))))))))) ((pw_214_0 i), ((pw_214_1 i), ((pw_214_2 i), ((pw_214_3 i), ((pw_214_4 i), ((pw_214_5 i), ((pw_214_6 i), ((pw_214_7 i), ()))))))))))
    (ph216_3 : (∀ i : Fin 4, Stage_preprocess.parallel_parallel_generatedRoot_407_218 Backend.backend model params' i0 i (pw_201_0, (pw_206_0, (pw_209_0, ()))) ((pw_216_0 i), ((pw_216_1 i), ((pw_216_2 i), ((pw_216_3 i), ()))))))
    (ph218_3 : (∀ i : Fin 4, Stage_preprocess.parallel_parallel_generatedRoot_407_220 Backend.backend model params' i0 i (pw_201_0, (pw_206_0, (pw_209_0, ()))) ((pw_218_0 i), ((pw_218_1 i), ((pw_218_2 i), ((pw_218_3 i), ()))))))
    {qw_4_0 : ExactMatrix Q0 8 1 1}
    {qw_11_0 : ExactMatrix Q0 8 1 1}
    {qw_293_0 : ExactMatrix Q7 8 1 12}
    {qw_294_0 : ExactMatrix Q7 8 1 12}
    {qw_295_0 : ExactMatrix Q7 8 1 12}
    {qw_296_0 : ExactMatrix Q7 8 1 12}
    {qw_307_0 : Int}
    {qw_309_0 : Int}
    {qw_325_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {qw_326_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {qw_327_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {qw_328_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {qw_329_0 : Fin 5 → Int}
    {qw_330_0 : Fin 5 → Int}
    {qw_332_0 : ByteArray}
    {qw_7_0 : ExactMatrix Q0 8 1 1}
    {qw_29_0 : ExactMatrix Q0 8 1 1}
    {qw_297_0 : ExactMatrix Q7 8 1 1}
    {qw_298_0 : ExactMatrix Q7 8 1 12}
    {qw_299_0 : ExactMatrix Q7 8 1 1}
    {qw_300_0 : ExactMatrix Q7 8 1 12}
    {qw_301_0 : ExactMatrix Q7 8 1 12}
    {qw_302_0 : ExactMatrix Q7 8 1 1}
    {qw_303_0 : ExactMatrix Q6 8 1 12}
    {qw_304_0 : ExactMatrix Q7 8 1 12}
    {qw_305_0 : ExactMatrix Q7 8 1 12}
    {qw_306_0 : ExactMatrix Q7 8 1 1}
    {qw_308_0 : ExactMatrix Q7 8 1 1}
    {qw_310_0 : ExactMatrix Q7 8 1 1}
    {qw_311_0 : ExactMatrix Q7 8 1 1}
    {qw_312_0 : ExactMatrix Q7 8 1 12}
    {qw_313_0 : ExactMatrix Q7 8 1 12}
    {qw_314_0 : ExactMatrix Q7 8 1 12}
    {qw_315_0 : ExactMatrix Q7 8 1 12}
    {qw_316_0 : ExactMatrix Q7 8 1 12}
    {qw_317_0 : ExactMatrix Q7 8 1 12}
    {qw_318_0 : ExactMatrix Q7 8 1 12}
    {qw_319_0 : ExactMatrix Q7 8 1 1}
    {qw_320_0 : ExactMatrix Q7 8 1 12}
    {qw_321_0 : ExactMatrix Q7 8 1 12}
    {qw_322_0 : ExactMatrix Q7 8 1 12}
    {qw_323_0 : ExactMatrix Q7 8 1 12}
    {qw_324_0 : ExactMatrix Q7 8 1 12}
    {qw_331_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {qw_331_1 : Fin 5 → ExactMatrix Q7 8 1 14}
    {qw_331_2 : Fin 5 → ExactMatrix Q7 8 1 14}
    {qw_331_3 : Fin 5 → ExactMatrix Q7 8 1 14}
    {qw_331_4 : Fin 5 → ExactMatrix Q7 8 1 14}
    {qw_331_5 : Fin 5 → ExactMatrix Q7 8 1 14}
    {qw_331_6 : Fin 5 → ExactMatrix Q7 8 1 1}
    {qw_335_0 : Fin 4 → ExactMatrix Q5 8 1 8}
    {qw_335_1 : Fin 4 → ExactMatrix Q5 8 1 8}
    {qw_336_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {qw_336_1 : Fin 4 → ExactMatrix Q0 8 1 6}
    (qe_7_0 : qw_7_0 = (1 : ExactMatrix Q0 8 1 1))
    (qe_29_0 : qw_29_0 = (1 : ExactMatrix Q0 8 1 1))
    (qe_298_0 : qw_298_0 = matrixMulScalarLeft qw_297_0 qw_293_0)
    (qe_300_0 : qw_300_0 = matrixMulScalarLeft qw_299_0 qw_295_0)
    (qe_301_0 : qw_301_0 = matrixAdd qw_298_0 qw_300_0)
    (qe_305_0 : qw_305_0 = matrixMulScalarLeft qw_302_0 qw_304_0)
    (qe_306_0 : qw_306_0 = (matrixPolynomial [(p6)] : ExactMatrix Q7 8 1 1))
    (qe_308_0 : qw_308_0 = liftInteger qw_307_0)
    (qe_310_0 : qw_310_0 = multiplyMonomial qw_308_0 qw_309_0)
    (qe_311_0 : qw_311_0 = matrixMulScalarLeft qw_306_0 qw_310_0)
    (qe_312_0 : qw_312_0 = matrixMulScalarRight qw_305_0 qw_311_0)
    (qe_313_0 : qw_313_0 = matrixSub qw_301_0 qw_312_0)
    (qe_315_0 : qw_315_0 = matrixAdd qw_313_0 qw_314_0)
    (qe_316_0 : qw_316_0 = matrixMulScalarLeft qw_297_0 qw_294_0)
    (qe_317_0 : qw_317_0 = matrixMulScalarLeft qw_299_0 qw_296_0)
    (qe_318_0 : qw_318_0 = matrixAdd qw_316_0 qw_317_0)
    (qe_320_0 : qw_320_0 = matrixMulScalarLeft qw_319_0 qw_304_0)
    (qe_321_0 : qw_321_0 = matrixMulScalarRight qw_320_0 qw_311_0)
    (qe_322_0 : qw_322_0 = matrixSub qw_318_0 qw_321_0)
    (qe_324_0 : qw_324_0 = matrixAdd qw_322_0 qw_323_0)
    (qh331_6 : (∀ i : Fin 5, Stage_encrypt.parallel_parallel_generatedRoot_454_331 Backend.backend model params'' i0 i ((qw_325_0 i), ((qw_326_0 i), ((qw_327_0 i), ((qw_328_0 i), (qw_297_0, (qw_299_0, ((qw_329_0 i), ((qw_330_0 i), ())))))))) ((qw_331_0 i), ((qw_331_1 i), ((qw_331_2 i), ((qw_331_3 i), ((qw_331_4 i), ((qw_331_5 i), ((qw_331_6 i), ())))))))))
    (qh335_1 : (∀ i : Fin 4, Stage_encrypt.parallel_parallel_generatedRoot_454_335 Backend.backend model params'' i0 i (qw_297_0, (qw_332_0, (qw_299_0, ()))) ((qw_335_0 i), ((qw_335_1 i), ()))))
    (qh336_1 : (∀ i : Fin 4, Stage_encrypt.parallel_parallel_generatedRoot_454_336 Backend.backend model params'' i0 i (qw_297_0, (qw_332_0, (qw_299_0, ()))) ((qw_336_0 i), ((qw_336_1 i), ()))))
    (qh297_0 : centeredRebaseRuns qw_4_0 qw_297_0)
    (qh299_0 : centeredRebaseRuns qw_7_0 qw_299_0)
    (qh302_0 : centeredRebaseRuns qw_11_0 qw_302_0)
    (qh303_0 : gadgetMatrixRuns Backend.backend 32768 12 qw_303_0)
    (qh304_0 : centeredRebaseRuns qw_303_0 qw_304_0)
    (qh314_0 : gaussianSample ((4/1)) (26) qw_314_0)
    (qh319_0 : centeredRebaseRuns qw_29_0 qw_319_0)
    (qh323_0 : gaussianSample ((4/1)) (26) qw_323_0)
    {r sm : ErrorPoly N} (hr : polyNorm r ≤ 1) (hsm : polyNorm sm ≤ 1)
    (hpay : pw_9_0 = fun _ _ ↦ reducePoly Q0 N r) (qhm : qw_4_0 = fun _ _ ↦ reducePoly Q0 N sm)
    (qhp : qw_11_0 = fun _ _ ↦ reducePoly Q0 N r) (q307 : qw_307_0 = 1) (q309 : qw_309_0 = 0)
        {GQ7_14 : ExactMatrix Q7 N 1 14} {GQ6_12 : ExactMatrix Q6 N 1 12} {RQ7_12 : ExactMatrix Q7 N 1 12}
    {GQ5_10 : ExactMatrix Q5 N 1 10} {RQ6_10 : ExactMatrix Q6 N 1 10}
    {GQ0_8 : ExactMatrix Q0 N 1 8} {RQ5_8 : ExactMatrix Q5 N 1 8}
    {GQ1_6 : ExactMatrix (R i0) N 1 6} {RQ0_6 : ExactMatrix Q0 N 1 6}
    (hGQ7_14 : gadgetMatrixRuns Backend.backend 32768 14 GQ7_14)
    (hGQ6_12 : gadgetMatrixRuns Backend.backend 32768 12 GQ6_12) (hRQ7_12 : centeredRebaseRuns GQ6_12 RQ7_12)
    (hGQ5_10 : gadgetMatrixRuns Backend.backend 32768 10 GQ5_10) (hRQ6_10 : centeredRebaseRuns GQ5_10 RQ6_10)
    (hGQ0_8 : gadgetMatrixRuns Backend.backend 32768 8 GQ0_8) (hRQ5_8 : centeredRebaseRuns GQ0_8 RQ5_8)
    (hGQ1_6 : gadgetMatrixRuns Backend.backend 32768 6 GQ1_6) (hRQ0_6 : centeredRebaseRuns GQ1_6 RQ0_6)
    (hlc : ∀ k, qw_329_0 k * qw_329_0 k = 1)
    (hlr : ∀ k, qw_330_0 k * 2 + (1 - qw_329_0 k) * 8 = (labelExp k : Int) * 4)
    (hle : ∀ k, 0 ≤ qw_330_0 k ∧ qw_330_0 k ≤ 7)
    (hs01 : ∀ i, 0 ≤ pw_207_0 i ∧ pw_207_0 i ≤ 1) (hs01' : ∀ i, 0 ≤ pw_212_0 i ∧ pw_212_0 i ≤ 1)
    (he0 : ∀ i, pw_208_0 i = 0) (he0' : ∀ i, pw_213_0 i = 0)
    (hsum : ∀ o : Fin 48, ∑ k : Fin 5, pw_207_0 (cellIndex o k) = 1)
    (hsum' : ∀ o : Fin 48, ∑ k : Fin 5, pw_212_0 (cellIndex o k) = 1) :
    Rel sm ![![qw_293_0, qw_294_0], ![qw_295_0, qw_296_0]]
      (fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) * 1 * unitPair r j) • RQ7_12) ![qw_315_0, qw_324_0] 26 ∧
    (∀ i : Fin 4, Rel sm (lutA model qw_332_0 Q5 8 2 (sigma i))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • RQ5_8) ![qw_335_0 i, qw_335_1 i] 26) ∧
    (∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q0 N (substitution (sigma i).toNat (sigma_odd i) r) •
          ![![pw_216_0, pw_216_1], ![pw_216_2, pw_216_3]] 0 j i + ![![pw_216_0, pw_216_1], ![pw_216_2, pw_216_3]] 1 j i =
        (lutCoefficient Q0 996917851236477127152305493547100497 (fun k ↦ reducePoly Q0 N (readoutTable k))
          (sigma i).toNat * reducePoly Q0 N (((((qd i0) : Nat) : Int) : ErrorPoly N) * unitPair r j)) • RQ0_6 +
            reduceMatrix Q0 N 1 6 (e j)) ∧
    (∀ i : Fin 4, Rel sm (lutA model qw_332_0 Q0 6 3 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • RQ0_6) ![qw_336_0 i, qw_336_1 i] 26) ∧
    (∀ k : Fin 5, Rel sm ![![qw_331_0 k, qw_331_1 k], ![qw_331_2 k, qw_331_3 k]]
      (fun j ↦ reducePoly Q7 N (Xe (n := N) (2 * labelExp k) * unitPair sm j) • GQ7_14)
      ![qw_331_4 k, qw_331_5 k] 26) ∧
    (∃ eBm : Fin 240 → Fin 2 → ErrorMatrix N 1 10, (∀ i j, CoeffBound (eBm i j) 26) ∧ ∀ (i : Fin 240) j,
      reducePoly Q6 N r • rhsTop model pw_201_0 Q6 10 (freshRhs0 i0) i j + ![pw_210_6 i, pw_210_7 i] j =
        reducePoly Q6 N (((p5 : Nat) : Int) * ((pw_207_0 i : Int) : ErrorPoly N) * unitPair r j) • RQ6_10 +
          reduceMatrix Q6 N 1 10 (eBm i j)) ∧
    (∃ eBm : Fin 240 → Fin 2 → ErrorMatrix N 1 8, (∀ i j, CoeffBound (eBm i j) 26) ∧ ∀ (i : Fin 240) j,
      reducePoly Q5 N r • rhsTop model pw_201_0 Q5 8 (freshRhs1 i0) i j + ![pw_214_6 i, pw_214_7 i] j =
        reducePoly Q5 N (((p4 : Nat) : Int) * ((pw_212_0 i : Int) : ErrorPoly N) * unitPair r j) • RQ5_8 +
          reduceMatrix Q5 N 1 8 (eBm i j)) ∧
    (∃ sel : Fin 48 → Fin 5, ∀ o k, pw_207_0 (cellIndex o k) = if k = sel o then 1 else 0) ∧
    (∃ sel : Fin 48 → Fin 5, ∀ o k, pw_212_0 (cellIndex o k) = if k = sel o then 1 else 0) ∧
    (∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) r) •
          ![![pw_218_0, pw_218_1], ![pw_218_2, pw_218_3]] 0 j i + ![![pw_218_0, pw_218_1], ![pw_218_2, pw_218_3]] 1 j i =
        (lutCoefficient (R i0) (inv4 i0) (fun k ↦ reducePoly (R i0) N
          (![((0 : Int) : ErrorPoly N), ((0 : Int) : ErrorPoly N), ((0 : Int) : ErrorPoly N),
            ((1 : Int) : ErrorPoly N)] k)) (sigma i).toNat *
          reducePoly (R i0) N (((1 : Int) : ErrorPoly N) * unitPair r j)) • GQ1_6 + reduceMatrix (R i0) N 1 6 (e j)) := by
  have hone : qw_29_0 = 1 := qe_29_0
  have h1 := encOne_315_0 (model := model) (e_7_0 := qe_7_0) (e_29_0 := qe_29_0) (e_298_0 := qe_298_0) (e_300_0 := qe_300_0) (e_301_0 := qe_301_0) (e_305_0 := qe_305_0) (e_306_0 := qe_306_0) (e_308_0 := qe_308_0) (e_310_0 := qe_310_0) (e_311_0 := qe_311_0) (e_312_0 := qe_312_0) (e_313_0 := qe_313_0) (e_315_0 := qe_315_0) (e_316_0 := qe_316_0) (e_317_0 := qe_317_0) (e_318_0 := qe_318_0) (e_320_0 := qe_320_0) (e_321_0 := qe_321_0) (e_322_0 := qe_322_0) (e_324_0 := qe_324_0) (h297_0 := qh297_0) (h299_0 := qh299_0) (h302_0 := qh302_0) (h303_0 := qh303_0) (h304_0 := qh304_0) (h314_0 := qh314_0) (h319_0 := qh319_0) (h323_0 := qh323_0) (hsm := hsm) (hsp := hr)
    (hsmIn := qhm) (hspIn := qhp) (hone := hone) (hg' := hGQ6_12) (hg := hRQ7_12)
  rw [q307, q309, message_one_zero] at h1
  have hsec : pw_206_0 = fun _ _ ↦ reducePoly Q7 N r :=
    rebase_small (q := Q0) (by inst_num) hr (by rw [← hpay]; exact ph206_0)
  have h34 : pw_209_0 = 1 := rebase_one (q := Q0) (by inst_num) (by rw [← pe_11_0]; exact ph209_0)
  have hsmQ7 : qw_297_0 = fun _ _ ↦ reducePoly Q7 N sm :=
    rebase_small (q := Q0) (by inst_num) hsm (by rw [← qhm]; exact qh297_0)
  have h343 : qw_299_0 = 1 := rebase_one (q := Q0) (by inst_num) (by rw [← qe_7_0]; exact qh299_0)
  have hread : (fun k ↦ reducePoly Q0 N (readoutTable k)) =
      fun k ↦ reducePoly Q0 N (![Xe (n := N) 2, Xe (n := N) 2, Xe (n := N) 6, Xe (n := N) 6] k) := by
    funext k; fin_cases k <;> rfl
  refine ⟨h1, fun i ↦ encHelper335 i (qh335_1 i) hsmQ7 h343 hGQ0_8 hRQ5_8, fun i ↦ ?_,
    fun i ↦ encHelper336 i (qh336_1 i) hsmQ7 h343 hGQ1_6 hRQ0_6,
    encLabelRel331 qh331_6 hsmQ7 h343 hGQ7_14 hlc hlr hle, ?_, ?_, ?_, ?_,
    fun i ↦ ?_⟩
  · obtain ⟨e, he, hs⟩ := preSwitch216 i (ph216_3 i) hsec h34 hGQ1_6 hRQ0_6
    refine ⟨e, he, fun j ↦ ?_⟩
    rw [hread]
    fin_cases j
    · cast_exact (hs 0)
    · cast_exact (hs 1)
  · have hc := fun i : Fin 240 ↦ preRhs210 (ph210_7 i) hsec h34 hGQ5_10 hRQ6_10
    choose eB heB hrel using hc
    refine ⟨eB, heB, fun i j ↦ ?_⟩
    have ht := preTop210 (ph210_7 i)
    have := hrel i j
    rw [ht, he0 i, message_zero] at this
    exact this
  · have hc := fun i : Fin 240 ↦ preRhs214 (ph214_7 i) hsec h34 hGQ0_8 hRQ5_8
    choose eB heB hrel using hc
    refine ⟨eB, heB, fun i j ↦ ?_⟩
    have ht := preTop214 (ph214_7 i)
    have := hrel i j
    rw [ht, he0' i, message_zero] at this
    exact this
  · choose sel hsel using fun o : Fin 48 ↦ onehot_of_sum (fun k ↦ hs01 (cellIndex o k)) (hsum o)
    exact ⟨sel, hsel⟩
  · choose sel hsel using fun o : Fin 48 ↦ onehot_of_sum (fun k ↦ hs01' (cellIndex o k)) (hsum' o)
    exact ⟨sel, hsel⟩
  · obtain ⟨e, he, hs⟩ := preSwitch218 i (ph218_3 i) hsec h34 hGQ1_6
    refine ⟨e, he, fun j ↦ ?_⟩
    fin_cases j
    · cast_exact (hs 0)
    · cast_exact (hs 1)

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The inputs of the smudge chain of decoder `0`. -/
theorem smudgeInputs0 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params' : Stage_preprocess.Params} {params'' : Stage_encrypt.Params}
    {pw_0_0 : ByteArray}
    {pw_9_0 : ExactMatrix Q0 8 1 1}
    {pw_18_0 : Fin 800 → Int}
    {pw_19_0 : Fin 800 → Int}
    {pw_22_0 : Fin 800 → Int}
    {pw_23_0 : Fin 800 → Int}
    {pw_10_0 : ExactMatrix Q6 8 1 1}
    {pw_11_0 : ExactMatrix Q0 8 1 1}
    {pw_12_0 : ExactMatrix Q6 8 1 1}
    {pw_13_0 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_13_1 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_13_2 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_13_3 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_14_0 : ExactMatrix Q6 8 1 10}
    {pw_15_0 : ExactMatrix Q6 8 1 10}
    {pw_16_0 : ExactMatrix Q6 8 1 10}
    {pw_17_0 : ExactMatrix Q6 8 1 10}
    {pw_20_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_20_1 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_20_2 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_20_3 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_20_4 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_20_5 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_20_6 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_20_7 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_21_0 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_21_1 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_21_2 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_21_3 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_24_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_24_1 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_24_2 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_24_3 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_24_4 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_24_5 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_24_6 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_24_7 : Fin 800 → ExactMatrix Q0 8 1 6}
    (pe_11_0 : pw_11_0 = (1 : ExactMatrix Q0 8 1 1))
    (ph10_0 : centeredRebaseRuns pw_9_0 pw_10_0)
    (ph12_0 : centeredRebaseRuns pw_11_0 pw_12_0)
    (ph20_7 : (∀ i : Fin 800, Stage_preprocess.parallel_parallel_generatedRoot_407_20 Backend.backend model params' i0 i (pw_0_0, (pw_14_0, (pw_15_0, (pw_16_0, (pw_17_0, (pw_10_0, ((pw_18_0 i), ((pw_19_0 i), (pw_12_0, ()))))))))) ((pw_20_0 i), ((pw_20_1 i), ((pw_20_2 i), ((pw_20_3 i), ((pw_20_4 i), ((pw_20_5 i), ((pw_20_6 i), ((pw_20_7 i), ()))))))))))
    (ph24_7 : (∀ i : Fin 800, Stage_preprocess.parallel_parallel_generatedRoot_407_24 Backend.backend model params' i0 i (pw_0_0, (pw_21_0, (pw_21_2, (pw_21_1, (pw_21_3, (pw_10_0, ((pw_22_0 i), ((pw_23_0 i), (pw_12_0, ()))))))))) ((pw_24_0 i), ((pw_24_1 i), ((pw_24_2 i), ((pw_24_3 i), ((pw_24_4 i), ((pw_24_5 i), ((pw_24_6 i), ((pw_24_7 i), ()))))))))))
    (ph13_3 : (∀ i : Fin 4, Stage_preprocess.parallel_parallel_generatedRoot_407_13 Backend.backend model params' i0 i (pw_0_0, (pw_10_0, (pw_12_0, ()))) ((pw_13_0 i), ((pw_13_1 i), ((pw_13_2 i), ((pw_13_3 i), ()))))))
    {qw_0_0 : ExactMatrix Q6 8 1 10}
    {qw_1_0 : ExactMatrix Q6 8 1 10}
    {qw_2_0 : ExactMatrix Q6 8 1 10}
    {qw_3_0 : ExactMatrix Q6 8 1 10}
    {qw_4_0 : ExactMatrix Q0 8 1 1}
    {qw_11_0 : ExactMatrix Q0 8 1 1}
    {qw_17_0 : Int}
    {qw_19_0 : Int}
    {qw_36_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_37_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_38_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_39_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_40_0 : Fin 5 → Int}
    {qw_41_0 : Fin 5 → Int}
    {qw_43_0 : ByteArray}
    {qw_5_0 : ExactMatrix Q6 8 1 1}
    {qw_6_0 : ExactMatrix Q6 8 1 10}
    {qw_7_0 : ExactMatrix Q0 8 1 1}
    {qw_8_0 : ExactMatrix Q6 8 1 1}
    {qw_9_0 : ExactMatrix Q6 8 1 10}
    {qw_10_0 : ExactMatrix Q6 8 1 10}
    {qw_12_0 : ExactMatrix Q6 8 1 1}
    {qw_13_0 : ExactMatrix Q5 8 1 10}
    {qw_14_0 : ExactMatrix Q6 8 1 10}
    {qw_15_0 : ExactMatrix Q6 8 1 10}
    {qw_16_0 : ExactMatrix Q6 8 1 1}
    {qw_18_0 : ExactMatrix Q6 8 1 1}
    {qw_20_0 : ExactMatrix Q6 8 1 1}
    {qw_21_0 : ExactMatrix Q6 8 1 1}
    {qw_22_0 : ExactMatrix Q6 8 1 10}
    {qw_23_0 : ExactMatrix Q6 8 1 10}
    {qw_24_0 : ExactMatrix Q6 8 1 10}
    {qw_25_0 : ExactMatrix Q6 8 1 10}
    {qw_26_0 : ExactMatrix Q6 8 1 10}
    {qw_27_0 : ExactMatrix Q6 8 1 10}
    {qw_28_0 : ExactMatrix Q6 8 1 10}
    {qw_29_0 : ExactMatrix Q0 8 1 1}
    {qw_30_0 : ExactMatrix Q6 8 1 1}
    {qw_31_0 : ExactMatrix Q6 8 1 10}
    {qw_32_0 : ExactMatrix Q6 8 1 10}
    {qw_33_0 : ExactMatrix Q6 8 1 10}
    {qw_34_0 : ExactMatrix Q6 8 1 10}
    {qw_35_0 : ExactMatrix Q6 8 1 10}
    {qw_42_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_42_1 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_42_2 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_42_3 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_42_4 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_42_5 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_42_6 : Fin 5 → ExactMatrix Q6 8 1 1}
    {qw_46_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {qw_46_1 : Fin 4 → ExactMatrix Q0 8 1 6}
    (qe_6_0 : qw_6_0 = matrixMulScalarLeft qw_5_0 qw_0_0)
    (qe_7_0 : qw_7_0 = (1 : ExactMatrix Q0 8 1 1))
    (qe_9_0 : qw_9_0 = matrixMulScalarLeft qw_8_0 qw_2_0)
    (qe_10_0 : qw_10_0 = matrixAdd qw_6_0 qw_9_0)
    (qe_15_0 : qw_15_0 = matrixMulScalarLeft qw_12_0 qw_14_0)
    (qe_16_0 : qw_16_0 = (matrixPolynomial [(p5)] : ExactMatrix Q6 8 1 1))
    (qe_18_0 : qw_18_0 = liftInteger qw_17_0)
    (qe_20_0 : qw_20_0 = multiplyMonomial qw_18_0 qw_19_0)
    (qe_21_0 : qw_21_0 = matrixMulScalarLeft qw_16_0 qw_20_0)
    (qe_22_0 : qw_22_0 = matrixMulScalarRight qw_15_0 qw_21_0)
    (qe_23_0 : qw_23_0 = matrixSub qw_10_0 qw_22_0)
    (qe_25_0 : qw_25_0 = matrixAdd qw_23_0 qw_24_0)
    (qe_26_0 : qw_26_0 = matrixMulScalarLeft qw_5_0 qw_1_0)
    (qe_27_0 : qw_27_0 = matrixMulScalarLeft qw_8_0 qw_3_0)
    (qe_28_0 : qw_28_0 = matrixAdd qw_26_0 qw_27_0)
    (qe_29_0 : qw_29_0 = (1 : ExactMatrix Q0 8 1 1))
    (qe_31_0 : qw_31_0 = matrixMulScalarLeft qw_30_0 qw_14_0)
    (qe_32_0 : qw_32_0 = matrixMulScalarRight qw_31_0 qw_21_0)
    (qe_33_0 : qw_33_0 = matrixSub qw_28_0 qw_32_0)
    (qe_35_0 : qw_35_0 = matrixAdd qw_33_0 qw_34_0)
    (qh46_1 : (∀ i : Fin 4, Stage_encrypt.parallel_parallel_generatedRoot_454_46 Backend.backend model params'' i0 i (qw_5_0, (qw_43_0, (qw_8_0, ()))) ((qw_46_0 i), ((qw_46_1 i), ()))))
    (qh42_6 : (∀ i : Fin 5, Stage_encrypt.parallel_parallel_generatedRoot_454_42 Backend.backend model params'' i0 i ((qw_36_0 i), ((qw_37_0 i), ((qw_38_0 i), ((qw_39_0 i), (qw_5_0, (qw_8_0, ((qw_40_0 i), ((qw_41_0 i), ())))))))) ((qw_42_0 i), ((qw_42_1 i), ((qw_42_2 i), ((qw_42_3 i), ((qw_42_4 i), ((qw_42_5 i), ((qw_42_6 i), ())))))))))
    (qh5_0 : centeredRebaseRuns qw_4_0 qw_5_0)
    (qh8_0 : centeredRebaseRuns qw_7_0 qw_8_0)
    (qh12_0 : centeredRebaseRuns qw_11_0 qw_12_0)
    (qh13_0 : gadgetMatrixRuns Backend.backend 32768 10 qw_13_0)
    (qh14_0 : centeredRebaseRuns qw_13_0 qw_14_0)
    (qh24_0 : gaussianSample ((4/1)) (26) qw_24_0)
    (qh30_0 : centeredRebaseRuns qw_29_0 qw_30_0)
    (qh34_0 : gaussianSample ((4/1)) (26) qw_34_0)
    {r sm : ErrorPoly N} (hr : polyNorm r ≤ 1) (hsm : polyNorm sm ≤ 1)
    (hpay : pw_9_0 = fun _ _ ↦ reducePoly Q0 N r) (qhm : qw_4_0 = fun _ _ ↦ reducePoly Q0 N sm)
    (qhp : qw_11_0 = fun _ _ ↦ reducePoly Q0 N r) (qm1 : qw_17_0 = 1) (qm2 : qw_19_0 = 0)
    {GQ6_12 : ExactMatrix Q6 N 1 12} {GQ5_10 : ExactMatrix Q5 N 1 10} {RQ6_10 : ExactMatrix Q6 N 1 10}
    {GQ0_8 : ExactMatrix Q0 N 1 8} {RQ5_8 : ExactMatrix Q5 N 1 8}
    {GQ1_6 : ExactMatrix (R i0) N 1 6} {RQ0_6 : ExactMatrix Q0 N 1 6}
    (hGQ6_12 : gadgetMatrixRuns Backend.backend 32768 12 GQ6_12)
    (hGQ5_10 : gadgetMatrixRuns Backend.backend 32768 10 GQ5_10) (hRQ6_10 : centeredRebaseRuns GQ5_10 RQ6_10)
    (hGQ0_8 : gadgetMatrixRuns Backend.backend 32768 8 GQ0_8) (hRQ5_8 : centeredRebaseRuns GQ0_8 RQ5_8)
    (hGQ1_6 : gadgetMatrixRuns Backend.backend 32768 6 GQ1_6) (hRQ0_6 : centeredRebaseRuns GQ1_6 RQ0_6)
    (hlc : ∀ k, qw_40_0 k * qw_40_0 k = 1)
    (hlr : ∀ k, qw_41_0 k * 2 + (1 - qw_40_0 k) * 8 = (labelExp k : Int) * 4)
    (hle : ∀ k, 0 ≤ qw_41_0 k ∧ qw_41_0 k ≤ 7)
    (hs01 : ∀ i, 0 ≤ pw_18_0 i ∧ pw_18_0 i ≤ 1) (hs01' : ∀ i, 0 ≤ pw_22_0 i ∧ pw_22_0 i ≤ 1)
    (he0 : ∀ i, pw_19_0 i = 0) (he0' : ∀ i, pw_23_0 i = 0)
    (hsum : ∀ o : Fin 160, ∑ k : Fin 5, pw_18_0 (cellIndex o k) = 1)
    (hsum' : ∀ o : Fin 160, ∑ k : Fin 5, pw_22_0 (cellIndex o k) = 1) :
    Rel sm ![![qw_0_0, qw_1_0], ![qw_2_0, qw_3_0]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair r j) • RQ6_10) ![qw_25_0, qw_35_0] 26 ∧
    (∀ i : Fin 4, Rel sm (lutA model qw_43_0 Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • RQ0_6) ![qw_46_0 i, qw_46_1 i] 26) ∧
    (∀ k : Fin 5, Rel sm ![![qw_42_0 k, qw_42_1 k], ![qw_42_2 k, qw_42_3 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp k) * unitPair sm j) • GQ6_12)
      ![qw_42_4 k, qw_42_5 k] 26) ∧
    (∃ eBm : Fin 800 → Fin 2 → ErrorMatrix N 1 8, (∀ i j, CoeffBound (eBm i j) 26) ∧ ∀ (i : Fin 800) j,
      reducePoly Q5 N r • rhsTop model pw_0_0 Q5 8 (smudge0Rhs0 i0) i j + ![pw_20_6 i, pw_20_7 i] j =
        reducePoly Q5 N (((p4 : Nat) : Int) * ((pw_18_0 i : Int) : ErrorPoly N) * unitPair r j) • RQ5_8 +
          reduceMatrix Q5 N 1 8 (eBm i j)) ∧
    (∃ eBm : Fin 800 → Fin 2 → ErrorMatrix N 1 6, (∀ i j, CoeffBound (eBm i j) 26) ∧ ∀ (i : Fin 800) j,
      reducePoly Q0 N r • rhsTop model pw_0_0 Q0 6 (smudge0Rhs1 i0) i j + ![pw_24_6 i, pw_24_7 i] j =
        reducePoly Q0 N ((((qd i0) : Nat) : Int) * ((pw_22_0 i : Int) : ErrorPoly N) * unitPair r j) • RQ0_6 +
          reduceMatrix Q0 N 1 6 (eBm i j)) ∧
    (∃ sel : Fin 160 → Fin 5, ∀ o k, pw_18_0 (cellIndex o k) = if k = sel o then 1 else 0) ∧
    (∃ sel : Fin 160 → Fin 5, ∀ o k, pw_22_0 (cellIndex o k) = if k = sel o then 1 else 0) ∧
    (∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) r) •
          ![![pw_13_0, pw_13_1], ![pw_13_2, pw_13_3]] 0 j i +
            ![![pw_13_0, pw_13_1], ![pw_13_2, pw_13_3]] 1 j i =
        (lutCoefficient (R i0) (inv4 i0) (fun k ↦ reducePoly (R i0) N (smudgeTable k))
          (sigma i).toNat * reducePoly (R i0) N ((1 : ErrorPoly N) * unitPair r j)) • GQ1_6 +
            reduceMatrix (R i0) N 1 6 (e j)) := by
  have hone : qw_29_0 = 1 := qe_29_0
  have h1 := encOne_25_0 (model := model) (e_6_0 := qe_6_0) (e_7_0 := qe_7_0) (e_9_0 := qe_9_0) (e_10_0 := qe_10_0) (e_15_0 := qe_15_0) (e_16_0 := qe_16_0) (e_18_0 := qe_18_0) (e_20_0 := qe_20_0) (e_21_0 := qe_21_0) (e_22_0 := qe_22_0) (e_23_0 := qe_23_0) (e_25_0 := qe_25_0) (e_26_0 := qe_26_0) (e_27_0 := qe_27_0) (e_28_0 := qe_28_0) (e_29_0 := qe_29_0) (e_31_0 := qe_31_0) (e_32_0 := qe_32_0) (e_33_0 := qe_33_0) (e_35_0 := qe_35_0) (h5_0 := qh5_0) (h8_0 := qh8_0) (h12_0 := qh12_0) (h13_0 := qh13_0) (h14_0 := qh14_0) (h24_0 := qh24_0) (h30_0 := qh30_0) (h34_0 := qh34_0) (hsm := hsm) (hsp := hr)
    (hsmIn := qhm) (hspIn := qhp) (hone := hone) (hg' := hGQ5_10) (hg := hRQ6_10)
  rw [qm1, qm2, message_one_zero] at h1
  have hsec : pw_10_0 = fun _ _ ↦ reducePoly Q6 N r :=
    rebase_small (q := Q0) (by inst_num) hr (by rw [← hpay]; exact ph10_0)
  have h1' : pw_12_0 = 1 := rebase_one (q := Q0) (by inst_num) (by rw [← pe_11_0]; exact ph12_0)
  have hsmQ6 : qw_5_0 = fun _ _ ↦ reducePoly Q6 N sm :=
    rebase_small (q := Q0) (by inst_num) hsm (by rw [← qhm]; exact qh5_0)
  have h343 : qw_8_0 = 1 := rebase_one (q := Q0) (by inst_num) (by rw [← qe_7_0]; exact qh8_0)
  have htab : (fun k ↦ reducePoly (R i0) N (smudgeTable k)) = fun k ↦ reducePoly (R i0) N
      (![((1 : Int) : ErrorPoly N), ((1 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N)] k) := by
    funext k; fin_cases k <;> rfl
  refine ⟨h1, fun i ↦ encHelper46 i (qh46_1 i) hsmQ6 h343 hGQ1_6 hRQ0_6,
    encLabelRel42 qh42_6 hsmQ6 h343 hGQ6_12 hlc hlr hle, ?_, ?_, ?_, ?_, fun i ↦ ?_⟩
  · have hc := fun i : Fin 800 ↦ preRhs20 (ph20_7 i) hsec h1' hGQ0_8 hRQ5_8
    choose eB heB hrel using hc
    refine ⟨eB, heB, fun i j ↦ ?_⟩
    have ht := preTop20 (ph20_7 i)
    have := hrel i j
    rw [ht, he0 i, message_zero] at this
    exact this
  · have hc := fun i : Fin 800 ↦ preRhs24 (ph24_7 i) hsec h1' hGQ1_6 hRQ0_6
    choose eB heB hrel using hc
    refine ⟨eB, heB, fun i j ↦ ?_⟩
    have ht := preTop24 (ph24_7 i)
    have := hrel i j
    rw [ht, he0' i, message_zero] at this
    exact this
  · choose sel hsel using fun o : Fin 160 ↦ onehot_of_sum (fun k ↦ hs01 (cellIndex o k)) (hsum o)
    exact ⟨sel, hsel⟩
  · choose sel hsel using fun o : Fin 160 ↦ onehot_of_sum (fun k ↦ hs01' (cellIndex o k)) (hsum' o)
    exact ⟨sel, hsel⟩
  · obtain ⟨e, he, hs⟩ := preSwitch13 i (ph13_3 i) hsec h1' hGQ1_6
    refine ⟨e, he, fun j ↦ ?_⟩
    rw [htab, show (1 : ErrorPoly N) = ((1 : Int) : ErrorPoly N) from Int.cast_one.symm]
    fin_cases j
    · cast_exact (hs 0)
    · cast_exact (hs 1)

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The inputs of the smudge chain of decoder `1`. -/
theorem smudgeInputs1 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params' : Stage_preprocess.Params} {params'' : Stage_encrypt.Params}
    {pw_9_0 : ExactMatrix Q0 8 1 1}
    {pw_363_0 : ByteArray}
    {pw_379_0 : Fin 800 → Int}
    {pw_380_0 : Fin 800 → Int}
    {pw_383_0 : Fin 800 → Int}
    {pw_384_0 : Fin 800 → Int}
    {pw_11_0 : ExactMatrix Q0 8 1 1}
    {pw_372_0 : ExactMatrix Q6 8 1 1}
    {pw_373_0 : ExactMatrix Q6 8 1 1}
    {pw_374_0 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_374_1 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_374_2 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_374_3 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_375_0 : ExactMatrix Q6 8 1 10}
    {pw_376_0 : ExactMatrix Q6 8 1 10}
    {pw_377_0 : ExactMatrix Q6 8 1 10}
    {pw_378_0 : ExactMatrix Q6 8 1 10}
    {pw_381_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_381_1 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_381_2 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_381_3 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_381_4 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_381_5 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_381_6 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_381_7 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_382_0 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_382_1 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_382_2 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_382_3 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_385_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_385_1 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_385_2 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_385_3 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_385_4 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_385_5 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_385_6 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_385_7 : Fin 800 → ExactMatrix Q0 8 1 6}
    (pe_11_0 : pw_11_0 = (1 : ExactMatrix Q0 8 1 1))
    (ph372_0 : centeredRebaseRuns pw_9_0 pw_372_0)
    (ph373_0 : centeredRebaseRuns pw_11_0 pw_373_0)
    (ph381_7 : (∀ i : Fin 800, Stage_preprocess.parallel_parallel_generatedRoot_407_385 Backend.backend model params' i0 i (pw_363_0, (pw_375_0, (pw_376_0, (pw_377_0, (pw_378_0, (pw_372_0, ((pw_379_0 i), ((pw_380_0 i), (pw_373_0, ()))))))))) ((pw_381_0 i), ((pw_381_1 i), ((pw_381_2 i), ((pw_381_3 i), ((pw_381_4 i), ((pw_381_5 i), ((pw_381_6 i), ((pw_381_7 i), ()))))))))))
    (ph385_7 : (∀ i : Fin 800, Stage_preprocess.parallel_parallel_generatedRoot_407_389 Backend.backend model params' i0 i (pw_363_0, (pw_382_0, (pw_382_2, (pw_382_1, (pw_382_3, (pw_372_0, ((pw_383_0 i), ((pw_384_0 i), (pw_373_0, ()))))))))) ((pw_385_0 i), ((pw_385_1 i), ((pw_385_2 i), ((pw_385_3 i), ((pw_385_4 i), ((pw_385_5 i), ((pw_385_6 i), ((pw_385_7 i), ()))))))))))
    (ph374_3 : (∀ i : Fin 4, Stage_preprocess.parallel_parallel_generatedRoot_407_378 Backend.backend model params' i0 i (pw_363_0, (pw_372_0, (pw_373_0, ()))) ((pw_374_0 i), ((pw_374_1 i), ((pw_374_2 i), ((pw_374_3 i), ()))))))
    {qw_4_0 : ExactMatrix Q0 8 1 1}
    {qw_11_0 : ExactMatrix Q0 8 1 1}
    {qw_47_0 : ExactMatrix Q6 8 1 10}
    {qw_48_0 : ExactMatrix Q6 8 1 10}
    {qw_49_0 : ExactMatrix Q6 8 1 10}
    {qw_50_0 : ExactMatrix Q6 8 1 10}
    {qw_61_0 : Int}
    {qw_63_0 : Int}
    {qw_79_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_80_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_81_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_82_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_83_0 : Fin 5 → Int}
    {qw_84_0 : Fin 5 → Int}
    {qw_86_0 : ByteArray}
    {qw_7_0 : ExactMatrix Q0 8 1 1}
    {qw_29_0 : ExactMatrix Q0 8 1 1}
    {qw_51_0 : ExactMatrix Q6 8 1 1}
    {qw_52_0 : ExactMatrix Q6 8 1 10}
    {qw_53_0 : ExactMatrix Q6 8 1 1}
    {qw_54_0 : ExactMatrix Q6 8 1 10}
    {qw_55_0 : ExactMatrix Q6 8 1 10}
    {qw_56_0 : ExactMatrix Q6 8 1 1}
    {qw_57_0 : ExactMatrix Q5 8 1 10}
    {qw_58_0 : ExactMatrix Q6 8 1 10}
    {qw_59_0 : ExactMatrix Q6 8 1 10}
    {qw_60_0 : ExactMatrix Q6 8 1 1}
    {qw_62_0 : ExactMatrix Q6 8 1 1}
    {qw_64_0 : ExactMatrix Q6 8 1 1}
    {qw_65_0 : ExactMatrix Q6 8 1 1}
    {qw_66_0 : ExactMatrix Q6 8 1 10}
    {qw_67_0 : ExactMatrix Q6 8 1 10}
    {qw_68_0 : ExactMatrix Q6 8 1 10}
    {qw_69_0 : ExactMatrix Q6 8 1 10}
    {qw_70_0 : ExactMatrix Q6 8 1 10}
    {qw_71_0 : ExactMatrix Q6 8 1 10}
    {qw_72_0 : ExactMatrix Q6 8 1 10}
    {qw_73_0 : ExactMatrix Q6 8 1 1}
    {qw_74_0 : ExactMatrix Q6 8 1 10}
    {qw_75_0 : ExactMatrix Q6 8 1 10}
    {qw_76_0 : ExactMatrix Q6 8 1 10}
    {qw_77_0 : ExactMatrix Q6 8 1 10}
    {qw_78_0 : ExactMatrix Q6 8 1 10}
    {qw_85_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_85_1 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_85_2 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_85_3 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_85_4 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_85_5 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_85_6 : Fin 5 → ExactMatrix Q6 8 1 1}
    {qw_89_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {qw_89_1 : Fin 4 → ExactMatrix Q0 8 1 6}
    (qe_7_0 : qw_7_0 = (1 : ExactMatrix Q0 8 1 1))
    (qe_29_0 : qw_29_0 = (1 : ExactMatrix Q0 8 1 1))
    (qe_52_0 : qw_52_0 = matrixMulScalarLeft qw_51_0 qw_47_0)
    (qe_54_0 : qw_54_0 = matrixMulScalarLeft qw_53_0 qw_49_0)
    (qe_55_0 : qw_55_0 = matrixAdd qw_52_0 qw_54_0)
    (qe_59_0 : qw_59_0 = matrixMulScalarLeft qw_56_0 qw_58_0)
    (qe_60_0 : qw_60_0 = (matrixPolynomial [(p5)] : ExactMatrix Q6 8 1 1))
    (qe_62_0 : qw_62_0 = liftInteger qw_61_0)
    (qe_64_0 : qw_64_0 = multiplyMonomial qw_62_0 qw_63_0)
    (qe_65_0 : qw_65_0 = matrixMulScalarLeft qw_60_0 qw_64_0)
    (qe_66_0 : qw_66_0 = matrixMulScalarRight qw_59_0 qw_65_0)
    (qe_67_0 : qw_67_0 = matrixSub qw_55_0 qw_66_0)
    (qe_69_0 : qw_69_0 = matrixAdd qw_67_0 qw_68_0)
    (qe_70_0 : qw_70_0 = matrixMulScalarLeft qw_51_0 qw_48_0)
    (qe_71_0 : qw_71_0 = matrixMulScalarLeft qw_53_0 qw_50_0)
    (qe_72_0 : qw_72_0 = matrixAdd qw_70_0 qw_71_0)
    (qe_74_0 : qw_74_0 = matrixMulScalarLeft qw_73_0 qw_58_0)
    (qe_75_0 : qw_75_0 = matrixMulScalarRight qw_74_0 qw_65_0)
    (qe_76_0 : qw_76_0 = matrixSub qw_72_0 qw_75_0)
    (qe_78_0 : qw_78_0 = matrixAdd qw_76_0 qw_77_0)
    (qh89_1 : (∀ i : Fin 4, Stage_encrypt.parallel_parallel_generatedRoot_454_89 Backend.backend model params'' i0 i (qw_51_0, (qw_86_0, (qw_53_0, ()))) ((qw_89_0 i), ((qw_89_1 i), ()))))
    (qh85_6 : (∀ i : Fin 5, Stage_encrypt.parallel_parallel_generatedRoot_454_85 Backend.backend model params'' i0 i ((qw_79_0 i), ((qw_80_0 i), ((qw_81_0 i), ((qw_82_0 i), (qw_51_0, (qw_53_0, ((qw_83_0 i), ((qw_84_0 i), ())))))))) ((qw_85_0 i), ((qw_85_1 i), ((qw_85_2 i), ((qw_85_3 i), ((qw_85_4 i), ((qw_85_5 i), ((qw_85_6 i), ())))))))))
    (qh51_0 : centeredRebaseRuns qw_4_0 qw_51_0)
    (qh53_0 : centeredRebaseRuns qw_7_0 qw_53_0)
    (qh56_0 : centeredRebaseRuns qw_11_0 qw_56_0)
    (qh57_0 : gadgetMatrixRuns Backend.backend 32768 10 qw_57_0)
    (qh58_0 : centeredRebaseRuns qw_57_0 qw_58_0)
    (qh68_0 : gaussianSample ((4/1)) (26) qw_68_0)
    (qh73_0 : centeredRebaseRuns qw_29_0 qw_73_0)
    (qh77_0 : gaussianSample ((4/1)) (26) qw_77_0)
    {r sm : ErrorPoly N} (hr : polyNorm r ≤ 1) (hsm : polyNorm sm ≤ 1)
    (hpay : pw_9_0 = fun _ _ ↦ reducePoly Q0 N r) (qhm : qw_4_0 = fun _ _ ↦ reducePoly Q0 N sm)
    (qhp : qw_11_0 = fun _ _ ↦ reducePoly Q0 N r) (qm1 : qw_61_0 = 1) (qm2 : qw_63_0 = 0)
    {GQ6_12 : ExactMatrix Q6 N 1 12} {GQ5_10 : ExactMatrix Q5 N 1 10} {RQ6_10 : ExactMatrix Q6 N 1 10}
    {GQ0_8 : ExactMatrix Q0 N 1 8} {RQ5_8 : ExactMatrix Q5 N 1 8}
    {GQ1_6 : ExactMatrix (R i0) N 1 6} {RQ0_6 : ExactMatrix Q0 N 1 6}
    (hGQ6_12 : gadgetMatrixRuns Backend.backend 32768 12 GQ6_12)
    (hGQ5_10 : gadgetMatrixRuns Backend.backend 32768 10 GQ5_10) (hRQ6_10 : centeredRebaseRuns GQ5_10 RQ6_10)
    (hGQ0_8 : gadgetMatrixRuns Backend.backend 32768 8 GQ0_8) (hRQ5_8 : centeredRebaseRuns GQ0_8 RQ5_8)
    (hGQ1_6 : gadgetMatrixRuns Backend.backend 32768 6 GQ1_6) (hRQ0_6 : centeredRebaseRuns GQ1_6 RQ0_6)
    (hlc : ∀ k, qw_83_0 k * qw_83_0 k = 1)
    (hlr : ∀ k, qw_84_0 k * 2 + (1 - qw_83_0 k) * 8 = (labelExp k : Int) * 4)
    (hle : ∀ k, 0 ≤ qw_84_0 k ∧ qw_84_0 k ≤ 7)
    (hs01 : ∀ i, 0 ≤ pw_379_0 i ∧ pw_379_0 i ≤ 1) (hs01' : ∀ i, 0 ≤ pw_383_0 i ∧ pw_383_0 i ≤ 1)
    (he0 : ∀ i, pw_380_0 i = 0) (he0' : ∀ i, pw_384_0 i = 0)
    (hsum : ∀ o : Fin 160, ∑ k : Fin 5, pw_379_0 (cellIndex o k) = 1)
    (hsum' : ∀ o : Fin 160, ∑ k : Fin 5, pw_383_0 (cellIndex o k) = 1) :
    Rel sm ![![qw_47_0, qw_48_0], ![qw_49_0, qw_50_0]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair r j) • RQ6_10) ![qw_69_0, qw_78_0] 26 ∧
    (∀ i : Fin 4, Rel sm (lutA model qw_86_0 Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • RQ0_6) ![qw_89_0 i, qw_89_1 i] 26) ∧
    (∀ k : Fin 5, Rel sm ![![qw_85_0 k, qw_85_1 k], ![qw_85_2 k, qw_85_3 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp k) * unitPair sm j) • GQ6_12)
      ![qw_85_4 k, qw_85_5 k] 26) ∧
    (∃ eBm : Fin 800 → Fin 2 → ErrorMatrix N 1 8, (∀ i j, CoeffBound (eBm i j) 26) ∧ ∀ (i : Fin 800) j,
      reducePoly Q5 N r • rhsTop model pw_363_0 Q5 8 (smudge1Rhs0 i0) i j + ![pw_381_6 i, pw_381_7 i] j =
        reducePoly Q5 N (((p4 : Nat) : Int) * ((pw_379_0 i : Int) : ErrorPoly N) * unitPair r j) • RQ5_8 +
          reduceMatrix Q5 N 1 8 (eBm i j)) ∧
    (∃ eBm : Fin 800 → Fin 2 → ErrorMatrix N 1 6, (∀ i j, CoeffBound (eBm i j) 26) ∧ ∀ (i : Fin 800) j,
      reducePoly Q0 N r • rhsTop model pw_363_0 Q0 6 (smudge1Rhs1 i0) i j + ![pw_385_6 i, pw_385_7 i] j =
        reducePoly Q0 N ((((qd i0) : Nat) : Int) * ((pw_383_0 i : Int) : ErrorPoly N) * unitPair r j) • RQ0_6 +
          reduceMatrix Q0 N 1 6 (eBm i j)) ∧
    (∃ sel : Fin 160 → Fin 5, ∀ o k, pw_379_0 (cellIndex o k) = if k = sel o then 1 else 0) ∧
    (∃ sel : Fin 160 → Fin 5, ∀ o k, pw_383_0 (cellIndex o k) = if k = sel o then 1 else 0) ∧
    (∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) r) •
          ![![pw_374_0, pw_374_1], ![pw_374_2, pw_374_3]] 0 j i +
            ![![pw_374_0, pw_374_1], ![pw_374_2, pw_374_3]] 1 j i =
        (lutCoefficient (R i0) (inv4 i0) (fun k ↦ reducePoly (R i0) N (smudgeTable k))
          (sigma i).toNat * reducePoly (R i0) N ((1 : ErrorPoly N) * unitPair r j)) • GQ1_6 +
            reduceMatrix (R i0) N 1 6 (e j)) := by
  have hone : qw_29_0 = 1 := qe_29_0
  have h1 := encOne_69_0 (model := model) (e_7_0 := qe_7_0) (e_29_0 := qe_29_0) (e_52_0 := qe_52_0) (e_54_0 := qe_54_0) (e_55_0 := qe_55_0) (e_59_0 := qe_59_0) (e_60_0 := qe_60_0) (e_62_0 := qe_62_0) (e_64_0 := qe_64_0) (e_65_0 := qe_65_0) (e_66_0 := qe_66_0) (e_67_0 := qe_67_0) (e_69_0 := qe_69_0) (e_70_0 := qe_70_0) (e_71_0 := qe_71_0) (e_72_0 := qe_72_0) (e_74_0 := qe_74_0) (e_75_0 := qe_75_0) (e_76_0 := qe_76_0) (e_78_0 := qe_78_0) (h51_0 := qh51_0) (h53_0 := qh53_0) (h56_0 := qh56_0) (h57_0 := qh57_0) (h58_0 := qh58_0) (h68_0 := qh68_0) (h73_0 := qh73_0) (h77_0 := qh77_0) (hsm := hsm) (hsp := hr)
    (hsmIn := qhm) (hspIn := qhp) (hone := hone) (hg' := hGQ5_10) (hg := hRQ6_10)
  rw [qm1, qm2, message_one_zero] at h1
  have hsec : pw_372_0 = fun _ _ ↦ reducePoly Q6 N r :=
    rebase_small (q := Q0) (by inst_num) hr (by rw [← hpay]; exact ph372_0)
  have h1' : pw_373_0 = 1 := rebase_one (q := Q0) (by inst_num) (by rw [← pe_11_0]; exact ph373_0)
  have hsmQ6 : qw_51_0 = fun _ _ ↦ reducePoly Q6 N sm :=
    rebase_small (q := Q0) (by inst_num) hsm (by rw [← qhm]; exact qh51_0)
  have h343 : qw_53_0 = 1 := rebase_one (q := Q0) (by inst_num) (by rw [← qe_7_0]; exact qh53_0)
  have htab : (fun k ↦ reducePoly (R i0) N (smudgeTable k)) = fun k ↦ reducePoly (R i0) N
      (![((1 : Int) : ErrorPoly N), ((1 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N)] k) := by
    funext k; fin_cases k <;> rfl
  refine ⟨h1, fun i ↦ encHelper89 i (qh89_1 i) hsmQ6 h343 hGQ1_6 hRQ0_6,
    encLabelRel85 qh85_6 hsmQ6 h343 hGQ6_12 hlc hlr hle, ?_, ?_, ?_, ?_, fun i ↦ ?_⟩
  · have hc := fun i : Fin 800 ↦ preRhs381 (ph381_7 i) hsec h1' hGQ0_8 hRQ5_8
    choose eB heB hrel using hc
    refine ⟨eB, heB, fun i j ↦ ?_⟩
    have ht := preTop381 (ph381_7 i)
    have := hrel i j
    rw [ht, he0 i, message_zero] at this
    exact this
  · have hc := fun i : Fin 800 ↦ preRhs385 (ph385_7 i) hsec h1' hGQ1_6 hRQ0_6
    choose eB heB hrel using hc
    refine ⟨eB, heB, fun i j ↦ ?_⟩
    have ht := preTop385 (ph385_7 i)
    have := hrel i j
    rw [ht, he0' i, message_zero] at this
    exact this
  · choose sel hsel using fun o : Fin 160 ↦ onehot_of_sum (fun k ↦ hs01 (cellIndex o k)) (hsum o)
    exact ⟨sel, hsel⟩
  · choose sel hsel using fun o : Fin 160 ↦ onehot_of_sum (fun k ↦ hs01' (cellIndex o k)) (hsum' o)
    exact ⟨sel, hsel⟩
  · obtain ⟨e, he, hs⟩ := preSwitch374 i (ph374_3 i) hsec h1' hGQ1_6
    refine ⟨e, he, fun j ↦ ?_⟩
    rw [htab, show (1 : ErrorPoly N) = ((1 : Int) : ErrorPoly N) from Int.cast_one.symm]
    fin_cases j
    · cast_exact (hs 0)
    · cast_exact (hs 1)

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The inputs of the smudge chain of decoder `2`. -/
theorem smudgeInputs2 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params' : Stage_preprocess.Params} {params'' : Stage_encrypt.Params}
    {pw_9_0 : ExactMatrix Q0 8 1 1}
    {pw_407_0 : ByteArray}
    {pw_423_0 : Fin 800 → Int}
    {pw_424_0 : Fin 800 → Int}
    {pw_427_0 : Fin 800 → Int}
    {pw_428_0 : Fin 800 → Int}
    {pw_11_0 : ExactMatrix Q0 8 1 1}
    {pw_416_0 : ExactMatrix Q6 8 1 1}
    {pw_417_0 : ExactMatrix Q6 8 1 1}
    {pw_418_0 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_418_1 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_418_2 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_418_3 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_419_0 : ExactMatrix Q6 8 1 10}
    {pw_420_0 : ExactMatrix Q6 8 1 10}
    {pw_421_0 : ExactMatrix Q6 8 1 10}
    {pw_422_0 : ExactMatrix Q6 8 1 10}
    {pw_425_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_425_1 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_425_2 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_425_3 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_425_4 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_425_5 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_425_6 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_425_7 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_426_0 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_426_1 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_426_2 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_426_3 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_429_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_429_1 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_429_2 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_429_3 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_429_4 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_429_5 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_429_6 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_429_7 : Fin 800 → ExactMatrix Q0 8 1 6}
    (pe_11_0 : pw_11_0 = (1 : ExactMatrix Q0 8 1 1))
    (ph416_0 : centeredRebaseRuns pw_9_0 pw_416_0)
    (ph417_0 : centeredRebaseRuns pw_11_0 pw_417_0)
    (ph425_7 : (∀ i : Fin 800, Stage_preprocess.parallel_parallel_generatedRoot_407_429 Backend.backend model params' i0 i (pw_407_0, (pw_419_0, (pw_420_0, (pw_421_0, (pw_422_0, (pw_416_0, ((pw_423_0 i), ((pw_424_0 i), (pw_417_0, ()))))))))) ((pw_425_0 i), ((pw_425_1 i), ((pw_425_2 i), ((pw_425_3 i), ((pw_425_4 i), ((pw_425_5 i), ((pw_425_6 i), ((pw_425_7 i), ()))))))))))
    (ph429_7 : (∀ i : Fin 800, Stage_preprocess.parallel_parallel_generatedRoot_407_433 Backend.backend model params' i0 i (pw_407_0, (pw_426_0, (pw_426_2, (pw_426_1, (pw_426_3, (pw_416_0, ((pw_427_0 i), ((pw_428_0 i), (pw_417_0, ()))))))))) ((pw_429_0 i), ((pw_429_1 i), ((pw_429_2 i), ((pw_429_3 i), ((pw_429_4 i), ((pw_429_5 i), ((pw_429_6 i), ((pw_429_7 i), ()))))))))))
    (ph418_3 : (∀ i : Fin 4, Stage_preprocess.parallel_parallel_generatedRoot_407_422 Backend.backend model params' i0 i (pw_407_0, (pw_416_0, (pw_417_0, ()))) ((pw_418_0 i), ((pw_418_1 i), ((pw_418_2 i), ((pw_418_3 i), ()))))))
    {qw_4_0 : ExactMatrix Q0 8 1 1}
    {qw_11_0 : ExactMatrix Q0 8 1 1}
    {qw_90_0 : ExactMatrix Q6 8 1 10}
    {qw_91_0 : ExactMatrix Q6 8 1 10}
    {qw_92_0 : ExactMatrix Q6 8 1 10}
    {qw_93_0 : ExactMatrix Q6 8 1 10}
    {qw_104_0 : Int}
    {qw_106_0 : Int}
    {qw_122_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_123_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_124_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_125_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_126_0 : Fin 5 → Int}
    {qw_127_0 : Fin 5 → Int}
    {qw_129_0 : ByteArray}
    {qw_7_0 : ExactMatrix Q0 8 1 1}
    {qw_29_0 : ExactMatrix Q0 8 1 1}
    {qw_94_0 : ExactMatrix Q6 8 1 1}
    {qw_95_0 : ExactMatrix Q6 8 1 10}
    {qw_96_0 : ExactMatrix Q6 8 1 1}
    {qw_97_0 : ExactMatrix Q6 8 1 10}
    {qw_98_0 : ExactMatrix Q6 8 1 10}
    {qw_99_0 : ExactMatrix Q6 8 1 1}
    {qw_100_0 : ExactMatrix Q5 8 1 10}
    {qw_101_0 : ExactMatrix Q6 8 1 10}
    {qw_102_0 : ExactMatrix Q6 8 1 10}
    {qw_103_0 : ExactMatrix Q6 8 1 1}
    {qw_105_0 : ExactMatrix Q6 8 1 1}
    {qw_107_0 : ExactMatrix Q6 8 1 1}
    {qw_108_0 : ExactMatrix Q6 8 1 1}
    {qw_109_0 : ExactMatrix Q6 8 1 10}
    {qw_110_0 : ExactMatrix Q6 8 1 10}
    {qw_111_0 : ExactMatrix Q6 8 1 10}
    {qw_112_0 : ExactMatrix Q6 8 1 10}
    {qw_113_0 : ExactMatrix Q6 8 1 10}
    {qw_114_0 : ExactMatrix Q6 8 1 10}
    {qw_115_0 : ExactMatrix Q6 8 1 10}
    {qw_116_0 : ExactMatrix Q6 8 1 1}
    {qw_117_0 : ExactMatrix Q6 8 1 10}
    {qw_118_0 : ExactMatrix Q6 8 1 10}
    {qw_119_0 : ExactMatrix Q6 8 1 10}
    {qw_120_0 : ExactMatrix Q6 8 1 10}
    {qw_121_0 : ExactMatrix Q6 8 1 10}
    {qw_128_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_128_1 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_128_2 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_128_3 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_128_4 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_128_5 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_128_6 : Fin 5 → ExactMatrix Q6 8 1 1}
    {qw_132_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {qw_132_1 : Fin 4 → ExactMatrix Q0 8 1 6}
    (qe_7_0 : qw_7_0 = (1 : ExactMatrix Q0 8 1 1))
    (qe_29_0 : qw_29_0 = (1 : ExactMatrix Q0 8 1 1))
    (qe_95_0 : qw_95_0 = matrixMulScalarLeft qw_94_0 qw_90_0)
    (qe_97_0 : qw_97_0 = matrixMulScalarLeft qw_96_0 qw_92_0)
    (qe_98_0 : qw_98_0 = matrixAdd qw_95_0 qw_97_0)
    (qe_102_0 : qw_102_0 = matrixMulScalarLeft qw_99_0 qw_101_0)
    (qe_103_0 : qw_103_0 = (matrixPolynomial [(p5)] : ExactMatrix Q6 8 1 1))
    (qe_105_0 : qw_105_0 = liftInteger qw_104_0)
    (qe_107_0 : qw_107_0 = multiplyMonomial qw_105_0 qw_106_0)
    (qe_108_0 : qw_108_0 = matrixMulScalarLeft qw_103_0 qw_107_0)
    (qe_109_0 : qw_109_0 = matrixMulScalarRight qw_102_0 qw_108_0)
    (qe_110_0 : qw_110_0 = matrixSub qw_98_0 qw_109_0)
    (qe_112_0 : qw_112_0 = matrixAdd qw_110_0 qw_111_0)
    (qe_113_0 : qw_113_0 = matrixMulScalarLeft qw_94_0 qw_91_0)
    (qe_114_0 : qw_114_0 = matrixMulScalarLeft qw_96_0 qw_93_0)
    (qe_115_0 : qw_115_0 = matrixAdd qw_113_0 qw_114_0)
    (qe_117_0 : qw_117_0 = matrixMulScalarLeft qw_116_0 qw_101_0)
    (qe_118_0 : qw_118_0 = matrixMulScalarRight qw_117_0 qw_108_0)
    (qe_119_0 : qw_119_0 = matrixSub qw_115_0 qw_118_0)
    (qe_121_0 : qw_121_0 = matrixAdd qw_119_0 qw_120_0)
    (qh132_1 : (∀ i : Fin 4, Stage_encrypt.parallel_parallel_generatedRoot_454_132 Backend.backend model params'' i0 i (qw_94_0, (qw_129_0, (qw_96_0, ()))) ((qw_132_0 i), ((qw_132_1 i), ()))))
    (qh128_6 : (∀ i : Fin 5, Stage_encrypt.parallel_parallel_generatedRoot_454_128 Backend.backend model params'' i0 i ((qw_122_0 i), ((qw_123_0 i), ((qw_124_0 i), ((qw_125_0 i), (qw_94_0, (qw_96_0, ((qw_126_0 i), ((qw_127_0 i), ())))))))) ((qw_128_0 i), ((qw_128_1 i), ((qw_128_2 i), ((qw_128_3 i), ((qw_128_4 i), ((qw_128_5 i), ((qw_128_6 i), ())))))))))
    (qh94_0 : centeredRebaseRuns qw_4_0 qw_94_0)
    (qh96_0 : centeredRebaseRuns qw_7_0 qw_96_0)
    (qh99_0 : centeredRebaseRuns qw_11_0 qw_99_0)
    (qh100_0 : gadgetMatrixRuns Backend.backend 32768 10 qw_100_0)
    (qh101_0 : centeredRebaseRuns qw_100_0 qw_101_0)
    (qh111_0 : gaussianSample ((4/1)) (26) qw_111_0)
    (qh116_0 : centeredRebaseRuns qw_29_0 qw_116_0)
    (qh120_0 : gaussianSample ((4/1)) (26) qw_120_0)
    {r sm : ErrorPoly N} (hr : polyNorm r ≤ 1) (hsm : polyNorm sm ≤ 1)
    (hpay : pw_9_0 = fun _ _ ↦ reducePoly Q0 N r) (qhm : qw_4_0 = fun _ _ ↦ reducePoly Q0 N sm)
    (qhp : qw_11_0 = fun _ _ ↦ reducePoly Q0 N r) (qm1 : qw_104_0 = 1) (qm2 : qw_106_0 = 0)
    {GQ6_12 : ExactMatrix Q6 N 1 12} {GQ5_10 : ExactMatrix Q5 N 1 10} {RQ6_10 : ExactMatrix Q6 N 1 10}
    {GQ0_8 : ExactMatrix Q0 N 1 8} {RQ5_8 : ExactMatrix Q5 N 1 8}
    {GQ1_6 : ExactMatrix (R i0) N 1 6} {RQ0_6 : ExactMatrix Q0 N 1 6}
    (hGQ6_12 : gadgetMatrixRuns Backend.backend 32768 12 GQ6_12)
    (hGQ5_10 : gadgetMatrixRuns Backend.backend 32768 10 GQ5_10) (hRQ6_10 : centeredRebaseRuns GQ5_10 RQ6_10)
    (hGQ0_8 : gadgetMatrixRuns Backend.backend 32768 8 GQ0_8) (hRQ5_8 : centeredRebaseRuns GQ0_8 RQ5_8)
    (hGQ1_6 : gadgetMatrixRuns Backend.backend 32768 6 GQ1_6) (hRQ0_6 : centeredRebaseRuns GQ1_6 RQ0_6)
    (hlc : ∀ k, qw_126_0 k * qw_126_0 k = 1)
    (hlr : ∀ k, qw_127_0 k * 2 + (1 - qw_126_0 k) * 8 = (labelExp k : Int) * 4)
    (hle : ∀ k, 0 ≤ qw_127_0 k ∧ qw_127_0 k ≤ 7)
    (hs01 : ∀ i, 0 ≤ pw_423_0 i ∧ pw_423_0 i ≤ 1) (hs01' : ∀ i, 0 ≤ pw_427_0 i ∧ pw_427_0 i ≤ 1)
    (he0 : ∀ i, pw_424_0 i = 0) (he0' : ∀ i, pw_428_0 i = 0)
    (hsum : ∀ o : Fin 160, ∑ k : Fin 5, pw_423_0 (cellIndex o k) = 1)
    (hsum' : ∀ o : Fin 160, ∑ k : Fin 5, pw_427_0 (cellIndex o k) = 1) :
    Rel sm ![![qw_90_0, qw_91_0], ![qw_92_0, qw_93_0]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair r j) • RQ6_10) ![qw_112_0, qw_121_0] 26 ∧
    (∀ i : Fin 4, Rel sm (lutA model qw_129_0 Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • RQ0_6) ![qw_132_0 i, qw_132_1 i] 26) ∧
    (∀ k : Fin 5, Rel sm ![![qw_128_0 k, qw_128_1 k], ![qw_128_2 k, qw_128_3 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp k) * unitPair sm j) • GQ6_12)
      ![qw_128_4 k, qw_128_5 k] 26) ∧
    (∃ eBm : Fin 800 → Fin 2 → ErrorMatrix N 1 8, (∀ i j, CoeffBound (eBm i j) 26) ∧ ∀ (i : Fin 800) j,
      reducePoly Q5 N r • rhsTop model pw_407_0 Q5 8 (smudge2Rhs0 i0) i j + ![pw_425_6 i, pw_425_7 i] j =
        reducePoly Q5 N (((p4 : Nat) : Int) * ((pw_423_0 i : Int) : ErrorPoly N) * unitPair r j) • RQ5_8 +
          reduceMatrix Q5 N 1 8 (eBm i j)) ∧
    (∃ eBm : Fin 800 → Fin 2 → ErrorMatrix N 1 6, (∀ i j, CoeffBound (eBm i j) 26) ∧ ∀ (i : Fin 800) j,
      reducePoly Q0 N r • rhsTop model pw_407_0 Q0 6 (smudge2Rhs1 i0) i j + ![pw_429_6 i, pw_429_7 i] j =
        reducePoly Q0 N ((((qd i0) : Nat) : Int) * ((pw_427_0 i : Int) : ErrorPoly N) * unitPair r j) • RQ0_6 +
          reduceMatrix Q0 N 1 6 (eBm i j)) ∧
    (∃ sel : Fin 160 → Fin 5, ∀ o k, pw_423_0 (cellIndex o k) = if k = sel o then 1 else 0) ∧
    (∃ sel : Fin 160 → Fin 5, ∀ o k, pw_427_0 (cellIndex o k) = if k = sel o then 1 else 0) ∧
    (∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) r) •
          ![![pw_418_0, pw_418_1], ![pw_418_2, pw_418_3]] 0 j i +
            ![![pw_418_0, pw_418_1], ![pw_418_2, pw_418_3]] 1 j i =
        (lutCoefficient (R i0) (inv4 i0) (fun k ↦ reducePoly (R i0) N (smudgeTable k))
          (sigma i).toNat * reducePoly (R i0) N ((1 : ErrorPoly N) * unitPair r j)) • GQ1_6 +
            reduceMatrix (R i0) N 1 6 (e j)) := by
  have hone : qw_29_0 = 1 := qe_29_0
  have h1 := encOne_112_0 (model := model) (e_7_0 := qe_7_0) (e_29_0 := qe_29_0) (e_95_0 := qe_95_0) (e_97_0 := qe_97_0) (e_98_0 := qe_98_0) (e_102_0 := qe_102_0) (e_103_0 := qe_103_0) (e_105_0 := qe_105_0) (e_107_0 := qe_107_0) (e_108_0 := qe_108_0) (e_109_0 := qe_109_0) (e_110_0 := qe_110_0) (e_112_0 := qe_112_0) (e_113_0 := qe_113_0) (e_114_0 := qe_114_0) (e_115_0 := qe_115_0) (e_117_0 := qe_117_0) (e_118_0 := qe_118_0) (e_119_0 := qe_119_0) (e_121_0 := qe_121_0) (h94_0 := qh94_0) (h96_0 := qh96_0) (h99_0 := qh99_0) (h100_0 := qh100_0) (h101_0 := qh101_0) (h111_0 := qh111_0) (h116_0 := qh116_0) (h120_0 := qh120_0) (hsm := hsm) (hsp := hr)
    (hsmIn := qhm) (hspIn := qhp) (hone := hone) (hg' := hGQ5_10) (hg := hRQ6_10)
  rw [qm1, qm2, message_one_zero] at h1
  have hsec : pw_416_0 = fun _ _ ↦ reducePoly Q6 N r :=
    rebase_small (q := Q0) (by inst_num) hr (by rw [← hpay]; exact ph416_0)
  have h1' : pw_417_0 = 1 := rebase_one (q := Q0) (by inst_num) (by rw [← pe_11_0]; exact ph417_0)
  have hsmQ6 : qw_94_0 = fun _ _ ↦ reducePoly Q6 N sm :=
    rebase_small (q := Q0) (by inst_num) hsm (by rw [← qhm]; exact qh94_0)
  have h343 : qw_96_0 = 1 := rebase_one (q := Q0) (by inst_num) (by rw [← qe_7_0]; exact qh96_0)
  have htab : (fun k ↦ reducePoly (R i0) N (smudgeTable k)) = fun k ↦ reducePoly (R i0) N
      (![((1 : Int) : ErrorPoly N), ((1 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N)] k) := by
    funext k; fin_cases k <;> rfl
  refine ⟨h1, fun i ↦ encHelper132 i (qh132_1 i) hsmQ6 h343 hGQ1_6 hRQ0_6,
    encLabelRel128 qh128_6 hsmQ6 h343 hGQ6_12 hlc hlr hle, ?_, ?_, ?_, ?_, fun i ↦ ?_⟩
  · have hc := fun i : Fin 800 ↦ preRhs425 (ph425_7 i) hsec h1' hGQ0_8 hRQ5_8
    choose eB heB hrel using hc
    refine ⟨eB, heB, fun i j ↦ ?_⟩
    have ht := preTop425 (ph425_7 i)
    have := hrel i j
    rw [ht, he0 i, message_zero] at this
    exact this
  · have hc := fun i : Fin 800 ↦ preRhs429 (ph429_7 i) hsec h1' hGQ1_6 hRQ0_6
    choose eB heB hrel using hc
    refine ⟨eB, heB, fun i j ↦ ?_⟩
    have ht := preTop429 (ph429_7 i)
    have := hrel i j
    rw [ht, he0' i, message_zero] at this
    exact this
  · choose sel hsel using fun o : Fin 160 ↦ onehot_of_sum (fun k ↦ hs01 (cellIndex o k)) (hsum o)
    exact ⟨sel, hsel⟩
  · choose sel hsel using fun o : Fin 160 ↦ onehot_of_sum (fun k ↦ hs01' (cellIndex o k)) (hsum' o)
    exact ⟨sel, hsel⟩
  · obtain ⟨e, he, hs⟩ := preSwitch418 i (ph418_3 i) hsec h1' hGQ1_6
    refine ⟨e, he, fun j ↦ ?_⟩
    rw [htab, show (1 : ErrorPoly N) = ((1 : Int) : ErrorPoly N) from Int.cast_one.symm]
    fin_cases j
    · cast_exact (hs 0)
    · cast_exact (hs 1)

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The inputs of the smudge chain of decoder `3`. -/
theorem smudgeInputs3 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params' : Stage_preprocess.Params} {params'' : Stage_encrypt.Params}
    {pw_9_0 : ExactMatrix Q0 8 1 1}
    {pw_451_0 : ByteArray}
    {pw_467_0 : Fin 800 → Int}
    {pw_468_0 : Fin 800 → Int}
    {pw_471_0 : Fin 800 → Int}
    {pw_472_0 : Fin 800 → Int}
    {pw_11_0 : ExactMatrix Q0 8 1 1}
    {pw_460_0 : ExactMatrix Q6 8 1 1}
    {pw_461_0 : ExactMatrix Q6 8 1 1}
    {pw_462_0 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_462_1 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_462_2 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_462_3 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_463_0 : ExactMatrix Q6 8 1 10}
    {pw_464_0 : ExactMatrix Q6 8 1 10}
    {pw_465_0 : ExactMatrix Q6 8 1 10}
    {pw_466_0 : ExactMatrix Q6 8 1 10}
    {pw_469_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_469_1 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_469_2 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_469_3 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_469_4 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_469_5 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_469_6 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_469_7 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_470_0 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_470_1 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_470_2 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_470_3 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_473_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_473_1 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_473_2 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_473_3 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_473_4 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_473_5 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_473_6 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_473_7 : Fin 800 → ExactMatrix Q0 8 1 6}
    (pe_11_0 : pw_11_0 = (1 : ExactMatrix Q0 8 1 1))
    (ph460_0 : centeredRebaseRuns pw_9_0 pw_460_0)
    (ph461_0 : centeredRebaseRuns pw_11_0 pw_461_0)
    (ph469_7 : (∀ i : Fin 800, Stage_preprocess.parallel_parallel_generatedRoot_407_473 Backend.backend model params' i0 i (pw_451_0, (pw_463_0, (pw_464_0, (pw_465_0, (pw_466_0, (pw_460_0, ((pw_467_0 i), ((pw_468_0 i), (pw_461_0, ()))))))))) ((pw_469_0 i), ((pw_469_1 i), ((pw_469_2 i), ((pw_469_3 i), ((pw_469_4 i), ((pw_469_5 i), ((pw_469_6 i), ((pw_469_7 i), ()))))))))))
    (ph473_7 : (∀ i : Fin 800, Stage_preprocess.parallel_parallel_generatedRoot_407_477 Backend.backend model params' i0 i (pw_451_0, (pw_470_0, (pw_470_2, (pw_470_1, (pw_470_3, (pw_460_0, ((pw_471_0 i), ((pw_472_0 i), (pw_461_0, ()))))))))) ((pw_473_0 i), ((pw_473_1 i), ((pw_473_2 i), ((pw_473_3 i), ((pw_473_4 i), ((pw_473_5 i), ((pw_473_6 i), ((pw_473_7 i), ()))))))))))
    (ph462_3 : (∀ i : Fin 4, Stage_preprocess.parallel_parallel_generatedRoot_407_466 Backend.backend model params' i0 i (pw_451_0, (pw_460_0, (pw_461_0, ()))) ((pw_462_0 i), ((pw_462_1 i), ((pw_462_2 i), ((pw_462_3 i), ()))))))
    {qw_4_0 : ExactMatrix Q0 8 1 1}
    {qw_11_0 : ExactMatrix Q0 8 1 1}
    {qw_133_0 : ExactMatrix Q6 8 1 10}
    {qw_134_0 : ExactMatrix Q6 8 1 10}
    {qw_135_0 : ExactMatrix Q6 8 1 10}
    {qw_136_0 : ExactMatrix Q6 8 1 10}
    {qw_147_0 : Int}
    {qw_149_0 : Int}
    {qw_165_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_166_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_167_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_168_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_169_0 : Fin 5 → Int}
    {qw_170_0 : Fin 5 → Int}
    {qw_172_0 : ByteArray}
    {qw_7_0 : ExactMatrix Q0 8 1 1}
    {qw_29_0 : ExactMatrix Q0 8 1 1}
    {qw_137_0 : ExactMatrix Q6 8 1 1}
    {qw_138_0 : ExactMatrix Q6 8 1 10}
    {qw_139_0 : ExactMatrix Q6 8 1 1}
    {qw_140_0 : ExactMatrix Q6 8 1 10}
    {qw_141_0 : ExactMatrix Q6 8 1 10}
    {qw_142_0 : ExactMatrix Q6 8 1 1}
    {qw_143_0 : ExactMatrix Q5 8 1 10}
    {qw_144_0 : ExactMatrix Q6 8 1 10}
    {qw_145_0 : ExactMatrix Q6 8 1 10}
    {qw_146_0 : ExactMatrix Q6 8 1 1}
    {qw_148_0 : ExactMatrix Q6 8 1 1}
    {qw_150_0 : ExactMatrix Q6 8 1 1}
    {qw_151_0 : ExactMatrix Q6 8 1 1}
    {qw_152_0 : ExactMatrix Q6 8 1 10}
    {qw_153_0 : ExactMatrix Q6 8 1 10}
    {qw_154_0 : ExactMatrix Q6 8 1 10}
    {qw_155_0 : ExactMatrix Q6 8 1 10}
    {qw_156_0 : ExactMatrix Q6 8 1 10}
    {qw_157_0 : ExactMatrix Q6 8 1 10}
    {qw_158_0 : ExactMatrix Q6 8 1 10}
    {qw_159_0 : ExactMatrix Q6 8 1 1}
    {qw_160_0 : ExactMatrix Q6 8 1 10}
    {qw_161_0 : ExactMatrix Q6 8 1 10}
    {qw_162_0 : ExactMatrix Q6 8 1 10}
    {qw_163_0 : ExactMatrix Q6 8 1 10}
    {qw_164_0 : ExactMatrix Q6 8 1 10}
    {qw_171_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_171_1 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_171_2 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_171_3 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_171_4 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_171_5 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_171_6 : Fin 5 → ExactMatrix Q6 8 1 1}
    {qw_175_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {qw_175_1 : Fin 4 → ExactMatrix Q0 8 1 6}
    (qe_7_0 : qw_7_0 = (1 : ExactMatrix Q0 8 1 1))
    (qe_29_0 : qw_29_0 = (1 : ExactMatrix Q0 8 1 1))
    (qe_138_0 : qw_138_0 = matrixMulScalarLeft qw_137_0 qw_133_0)
    (qe_140_0 : qw_140_0 = matrixMulScalarLeft qw_139_0 qw_135_0)
    (qe_141_0 : qw_141_0 = matrixAdd qw_138_0 qw_140_0)
    (qe_145_0 : qw_145_0 = matrixMulScalarLeft qw_142_0 qw_144_0)
    (qe_146_0 : qw_146_0 = (matrixPolynomial [(p5)] : ExactMatrix Q6 8 1 1))
    (qe_148_0 : qw_148_0 = liftInteger qw_147_0)
    (qe_150_0 : qw_150_0 = multiplyMonomial qw_148_0 qw_149_0)
    (qe_151_0 : qw_151_0 = matrixMulScalarLeft qw_146_0 qw_150_0)
    (qe_152_0 : qw_152_0 = matrixMulScalarRight qw_145_0 qw_151_0)
    (qe_153_0 : qw_153_0 = matrixSub qw_141_0 qw_152_0)
    (qe_155_0 : qw_155_0 = matrixAdd qw_153_0 qw_154_0)
    (qe_156_0 : qw_156_0 = matrixMulScalarLeft qw_137_0 qw_134_0)
    (qe_157_0 : qw_157_0 = matrixMulScalarLeft qw_139_0 qw_136_0)
    (qe_158_0 : qw_158_0 = matrixAdd qw_156_0 qw_157_0)
    (qe_160_0 : qw_160_0 = matrixMulScalarLeft qw_159_0 qw_144_0)
    (qe_161_0 : qw_161_0 = matrixMulScalarRight qw_160_0 qw_151_0)
    (qe_162_0 : qw_162_0 = matrixSub qw_158_0 qw_161_0)
    (qe_164_0 : qw_164_0 = matrixAdd qw_162_0 qw_163_0)
    (qh175_1 : (∀ i : Fin 4, Stage_encrypt.parallel_parallel_generatedRoot_454_175 Backend.backend model params'' i0 i (qw_137_0, (qw_172_0, (qw_139_0, ()))) ((qw_175_0 i), ((qw_175_1 i), ()))))
    (qh171_6 : (∀ i : Fin 5, Stage_encrypt.parallel_parallel_generatedRoot_454_171 Backend.backend model params'' i0 i ((qw_165_0 i), ((qw_166_0 i), ((qw_167_0 i), ((qw_168_0 i), (qw_137_0, (qw_139_0, ((qw_169_0 i), ((qw_170_0 i), ())))))))) ((qw_171_0 i), ((qw_171_1 i), ((qw_171_2 i), ((qw_171_3 i), ((qw_171_4 i), ((qw_171_5 i), ((qw_171_6 i), ())))))))))
    (qh137_0 : centeredRebaseRuns qw_4_0 qw_137_0)
    (qh139_0 : centeredRebaseRuns qw_7_0 qw_139_0)
    (qh142_0 : centeredRebaseRuns qw_11_0 qw_142_0)
    (qh143_0 : gadgetMatrixRuns Backend.backend 32768 10 qw_143_0)
    (qh144_0 : centeredRebaseRuns qw_143_0 qw_144_0)
    (qh154_0 : gaussianSample ((4/1)) (26) qw_154_0)
    (qh159_0 : centeredRebaseRuns qw_29_0 qw_159_0)
    (qh163_0 : gaussianSample ((4/1)) (26) qw_163_0)
    {r sm : ErrorPoly N} (hr : polyNorm r ≤ 1) (hsm : polyNorm sm ≤ 1)
    (hpay : pw_9_0 = fun _ _ ↦ reducePoly Q0 N r) (qhm : qw_4_0 = fun _ _ ↦ reducePoly Q0 N sm)
    (qhp : qw_11_0 = fun _ _ ↦ reducePoly Q0 N r) (qm1 : qw_147_0 = 1) (qm2 : qw_149_0 = 0)
    {GQ6_12 : ExactMatrix Q6 N 1 12} {GQ5_10 : ExactMatrix Q5 N 1 10} {RQ6_10 : ExactMatrix Q6 N 1 10}
    {GQ0_8 : ExactMatrix Q0 N 1 8} {RQ5_8 : ExactMatrix Q5 N 1 8}
    {GQ1_6 : ExactMatrix (R i0) N 1 6} {RQ0_6 : ExactMatrix Q0 N 1 6}
    (hGQ6_12 : gadgetMatrixRuns Backend.backend 32768 12 GQ6_12)
    (hGQ5_10 : gadgetMatrixRuns Backend.backend 32768 10 GQ5_10) (hRQ6_10 : centeredRebaseRuns GQ5_10 RQ6_10)
    (hGQ0_8 : gadgetMatrixRuns Backend.backend 32768 8 GQ0_8) (hRQ5_8 : centeredRebaseRuns GQ0_8 RQ5_8)
    (hGQ1_6 : gadgetMatrixRuns Backend.backend 32768 6 GQ1_6) (hRQ0_6 : centeredRebaseRuns GQ1_6 RQ0_6)
    (hlc : ∀ k, qw_169_0 k * qw_169_0 k = 1)
    (hlr : ∀ k, qw_170_0 k * 2 + (1 - qw_169_0 k) * 8 = (labelExp k : Int) * 4)
    (hle : ∀ k, 0 ≤ qw_170_0 k ∧ qw_170_0 k ≤ 7)
    (hs01 : ∀ i, 0 ≤ pw_467_0 i ∧ pw_467_0 i ≤ 1) (hs01' : ∀ i, 0 ≤ pw_471_0 i ∧ pw_471_0 i ≤ 1)
    (he0 : ∀ i, pw_468_0 i = 0) (he0' : ∀ i, pw_472_0 i = 0)
    (hsum : ∀ o : Fin 160, ∑ k : Fin 5, pw_467_0 (cellIndex o k) = 1)
    (hsum' : ∀ o : Fin 160, ∑ k : Fin 5, pw_471_0 (cellIndex o k) = 1) :
    Rel sm ![![qw_133_0, qw_134_0], ![qw_135_0, qw_136_0]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair r j) • RQ6_10) ![qw_155_0, qw_164_0] 26 ∧
    (∀ i : Fin 4, Rel sm (lutA model qw_172_0 Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • RQ0_6) ![qw_175_0 i, qw_175_1 i] 26) ∧
    (∀ k : Fin 5, Rel sm ![![qw_171_0 k, qw_171_1 k], ![qw_171_2 k, qw_171_3 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp k) * unitPair sm j) • GQ6_12)
      ![qw_171_4 k, qw_171_5 k] 26) ∧
    (∃ eBm : Fin 800 → Fin 2 → ErrorMatrix N 1 8, (∀ i j, CoeffBound (eBm i j) 26) ∧ ∀ (i : Fin 800) j,
      reducePoly Q5 N r • rhsTop model pw_451_0 Q5 8 (smudge3Rhs0 i0) i j + ![pw_469_6 i, pw_469_7 i] j =
        reducePoly Q5 N (((p4 : Nat) : Int) * ((pw_467_0 i : Int) : ErrorPoly N) * unitPair r j) • RQ5_8 +
          reduceMatrix Q5 N 1 8 (eBm i j)) ∧
    (∃ eBm : Fin 800 → Fin 2 → ErrorMatrix N 1 6, (∀ i j, CoeffBound (eBm i j) 26) ∧ ∀ (i : Fin 800) j,
      reducePoly Q0 N r • rhsTop model pw_451_0 Q0 6 (smudge3Rhs1 i0) i j + ![pw_473_6 i, pw_473_7 i] j =
        reducePoly Q0 N ((((qd i0) : Nat) : Int) * ((pw_471_0 i : Int) : ErrorPoly N) * unitPair r j) • RQ0_6 +
          reduceMatrix Q0 N 1 6 (eBm i j)) ∧
    (∃ sel : Fin 160 → Fin 5, ∀ o k, pw_467_0 (cellIndex o k) = if k = sel o then 1 else 0) ∧
    (∃ sel : Fin 160 → Fin 5, ∀ o k, pw_471_0 (cellIndex o k) = if k = sel o then 1 else 0) ∧
    (∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) r) •
          ![![pw_462_0, pw_462_1], ![pw_462_2, pw_462_3]] 0 j i +
            ![![pw_462_0, pw_462_1], ![pw_462_2, pw_462_3]] 1 j i =
        (lutCoefficient (R i0) (inv4 i0) (fun k ↦ reducePoly (R i0) N (smudgeTable k))
          (sigma i).toNat * reducePoly (R i0) N ((1 : ErrorPoly N) * unitPair r j)) • GQ1_6 +
            reduceMatrix (R i0) N 1 6 (e j)) := by
  have hone : qw_29_0 = 1 := qe_29_0
  have h1 := encOne_155_0 (model := model) (e_7_0 := qe_7_0) (e_29_0 := qe_29_0) (e_138_0 := qe_138_0) (e_140_0 := qe_140_0) (e_141_0 := qe_141_0) (e_145_0 := qe_145_0) (e_146_0 := qe_146_0) (e_148_0 := qe_148_0) (e_150_0 := qe_150_0) (e_151_0 := qe_151_0) (e_152_0 := qe_152_0) (e_153_0 := qe_153_0) (e_155_0 := qe_155_0) (e_156_0 := qe_156_0) (e_157_0 := qe_157_0) (e_158_0 := qe_158_0) (e_160_0 := qe_160_0) (e_161_0 := qe_161_0) (e_162_0 := qe_162_0) (e_164_0 := qe_164_0) (h137_0 := qh137_0) (h139_0 := qh139_0) (h142_0 := qh142_0) (h143_0 := qh143_0) (h144_0 := qh144_0) (h154_0 := qh154_0) (h159_0 := qh159_0) (h163_0 := qh163_0) (hsm := hsm) (hsp := hr)
    (hsmIn := qhm) (hspIn := qhp) (hone := hone) (hg' := hGQ5_10) (hg := hRQ6_10)
  rw [qm1, qm2, message_one_zero] at h1
  have hsec : pw_460_0 = fun _ _ ↦ reducePoly Q6 N r :=
    rebase_small (q := Q0) (by inst_num) hr (by rw [← hpay]; exact ph460_0)
  have h1' : pw_461_0 = 1 := rebase_one (q := Q0) (by inst_num) (by rw [← pe_11_0]; exact ph461_0)
  have hsmQ6 : qw_137_0 = fun _ _ ↦ reducePoly Q6 N sm :=
    rebase_small (q := Q0) (by inst_num) hsm (by rw [← qhm]; exact qh137_0)
  have h343 : qw_139_0 = 1 := rebase_one (q := Q0) (by inst_num) (by rw [← qe_7_0]; exact qh139_0)
  have htab : (fun k ↦ reducePoly (R i0) N (smudgeTable k)) = fun k ↦ reducePoly (R i0) N
      (![((1 : Int) : ErrorPoly N), ((1 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N)] k) := by
    funext k; fin_cases k <;> rfl
  refine ⟨h1, fun i ↦ encHelper175 i (qh175_1 i) hsmQ6 h343 hGQ1_6 hRQ0_6,
    encLabelRel171 qh171_6 hsmQ6 h343 hGQ6_12 hlc hlr hle, ?_, ?_, ?_, ?_, fun i ↦ ?_⟩
  · have hc := fun i : Fin 800 ↦ preRhs469 (ph469_7 i) hsec h1' hGQ0_8 hRQ5_8
    choose eB heB hrel using hc
    refine ⟨eB, heB, fun i j ↦ ?_⟩
    have ht := preTop469 (ph469_7 i)
    have := hrel i j
    rw [ht, he0 i, message_zero] at this
    exact this
  · have hc := fun i : Fin 800 ↦ preRhs473 (ph473_7 i) hsec h1' hGQ1_6 hRQ0_6
    choose eB heB hrel using hc
    refine ⟨eB, heB, fun i j ↦ ?_⟩
    have ht := preTop473 (ph473_7 i)
    have := hrel i j
    rw [ht, he0' i, message_zero] at this
    exact this
  · choose sel hsel using fun o : Fin 160 ↦ onehot_of_sum (fun k ↦ hs01 (cellIndex o k)) (hsum o)
    exact ⟨sel, hsel⟩
  · choose sel hsel using fun o : Fin 160 ↦ onehot_of_sum (fun k ↦ hs01' (cellIndex o k)) (hsum' o)
    exact ⟨sel, hsel⟩
  · obtain ⟨e, he, hs⟩ := preSwitch462 i (ph462_3 i) hsec h1' hGQ1_6
    refine ⟨e, he, fun j ↦ ?_⟩
    rw [htab, show (1 : ErrorPoly N) = ((1 : Int) : ErrorPoly N) from Int.cast_one.symm]
    fin_cases j
    · cast_exact (hs 0)
    · cast_exact (hs 1)

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The inputs of the smudge chain of decoder `4`. -/
theorem smudgeInputs4 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params' : Stage_preprocess.Params} {params'' : Stage_encrypt.Params}
    {pw_9_0 : ExactMatrix Q0 8 1 1}
    {pw_495_0 : ByteArray}
    {pw_511_0 : Fin 800 → Int}
    {pw_512_0 : Fin 800 → Int}
    {pw_515_0 : Fin 800 → Int}
    {pw_516_0 : Fin 800 → Int}
    {pw_11_0 : ExactMatrix Q0 8 1 1}
    {pw_504_0 : ExactMatrix Q6 8 1 1}
    {pw_505_0 : ExactMatrix Q6 8 1 1}
    {pw_506_0 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_506_1 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_506_2 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_506_3 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_507_0 : ExactMatrix Q6 8 1 10}
    {pw_508_0 : ExactMatrix Q6 8 1 10}
    {pw_509_0 : ExactMatrix Q6 8 1 10}
    {pw_510_0 : ExactMatrix Q6 8 1 10}
    {pw_513_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_513_1 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_513_2 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_513_3 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_513_4 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_513_5 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_513_6 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_513_7 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_514_0 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_514_1 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_514_2 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_514_3 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_517_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_517_1 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_517_2 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_517_3 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_517_4 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_517_5 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_517_6 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_517_7 : Fin 800 → ExactMatrix Q0 8 1 6}
    (pe_11_0 : pw_11_0 = (1 : ExactMatrix Q0 8 1 1))
    (ph504_0 : centeredRebaseRuns pw_9_0 pw_504_0)
    (ph505_0 : centeredRebaseRuns pw_11_0 pw_505_0)
    (ph513_7 : (∀ i : Fin 800, Stage_preprocess.parallel_parallel_generatedRoot_407_517 Backend.backend model params' i0 i (pw_495_0, (pw_507_0, (pw_508_0, (pw_509_0, (pw_510_0, (pw_504_0, ((pw_511_0 i), ((pw_512_0 i), (pw_505_0, ()))))))))) ((pw_513_0 i), ((pw_513_1 i), ((pw_513_2 i), ((pw_513_3 i), ((pw_513_4 i), ((pw_513_5 i), ((pw_513_6 i), ((pw_513_7 i), ()))))))))))
    (ph517_7 : (∀ i : Fin 800, Stage_preprocess.parallel_parallel_generatedRoot_407_521 Backend.backend model params' i0 i (pw_495_0, (pw_514_0, (pw_514_2, (pw_514_1, (pw_514_3, (pw_504_0, ((pw_515_0 i), ((pw_516_0 i), (pw_505_0, ()))))))))) ((pw_517_0 i), ((pw_517_1 i), ((pw_517_2 i), ((pw_517_3 i), ((pw_517_4 i), ((pw_517_5 i), ((pw_517_6 i), ((pw_517_7 i), ()))))))))))
    (ph506_3 : (∀ i : Fin 4, Stage_preprocess.parallel_parallel_generatedRoot_407_510 Backend.backend model params' i0 i (pw_495_0, (pw_504_0, (pw_505_0, ()))) ((pw_506_0 i), ((pw_506_1 i), ((pw_506_2 i), ((pw_506_3 i), ()))))))
    {qw_4_0 : ExactMatrix Q0 8 1 1}
    {qw_11_0 : ExactMatrix Q0 8 1 1}
    {qw_176_0 : ExactMatrix Q6 8 1 10}
    {qw_177_0 : ExactMatrix Q6 8 1 10}
    {qw_178_0 : ExactMatrix Q6 8 1 10}
    {qw_179_0 : ExactMatrix Q6 8 1 10}
    {qw_190_0 : Int}
    {qw_192_0 : Int}
    {qw_208_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_209_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_210_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_211_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_212_0 : Fin 5 → Int}
    {qw_213_0 : Fin 5 → Int}
    {qw_215_0 : ByteArray}
    {qw_7_0 : ExactMatrix Q0 8 1 1}
    {qw_29_0 : ExactMatrix Q0 8 1 1}
    {qw_180_0 : ExactMatrix Q6 8 1 1}
    {qw_181_0 : ExactMatrix Q6 8 1 10}
    {qw_182_0 : ExactMatrix Q6 8 1 1}
    {qw_183_0 : ExactMatrix Q6 8 1 10}
    {qw_184_0 : ExactMatrix Q6 8 1 10}
    {qw_185_0 : ExactMatrix Q6 8 1 1}
    {qw_186_0 : ExactMatrix Q5 8 1 10}
    {qw_187_0 : ExactMatrix Q6 8 1 10}
    {qw_188_0 : ExactMatrix Q6 8 1 10}
    {qw_189_0 : ExactMatrix Q6 8 1 1}
    {qw_191_0 : ExactMatrix Q6 8 1 1}
    {qw_193_0 : ExactMatrix Q6 8 1 1}
    {qw_194_0 : ExactMatrix Q6 8 1 1}
    {qw_195_0 : ExactMatrix Q6 8 1 10}
    {qw_196_0 : ExactMatrix Q6 8 1 10}
    {qw_197_0 : ExactMatrix Q6 8 1 10}
    {qw_198_0 : ExactMatrix Q6 8 1 10}
    {qw_199_0 : ExactMatrix Q6 8 1 10}
    {qw_200_0 : ExactMatrix Q6 8 1 10}
    {qw_201_0 : ExactMatrix Q6 8 1 10}
    {qw_202_0 : ExactMatrix Q6 8 1 1}
    {qw_203_0 : ExactMatrix Q6 8 1 10}
    {qw_204_0 : ExactMatrix Q6 8 1 10}
    {qw_205_0 : ExactMatrix Q6 8 1 10}
    {qw_206_0 : ExactMatrix Q6 8 1 10}
    {qw_207_0 : ExactMatrix Q6 8 1 10}
    {qw_214_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_214_1 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_214_2 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_214_3 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_214_4 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_214_5 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_214_6 : Fin 5 → ExactMatrix Q6 8 1 1}
    {qw_218_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {qw_218_1 : Fin 4 → ExactMatrix Q0 8 1 6}
    (qe_7_0 : qw_7_0 = (1 : ExactMatrix Q0 8 1 1))
    (qe_29_0 : qw_29_0 = (1 : ExactMatrix Q0 8 1 1))
    (qe_181_0 : qw_181_0 = matrixMulScalarLeft qw_180_0 qw_176_0)
    (qe_183_0 : qw_183_0 = matrixMulScalarLeft qw_182_0 qw_178_0)
    (qe_184_0 : qw_184_0 = matrixAdd qw_181_0 qw_183_0)
    (qe_188_0 : qw_188_0 = matrixMulScalarLeft qw_185_0 qw_187_0)
    (qe_189_0 : qw_189_0 = (matrixPolynomial [(p5)] : ExactMatrix Q6 8 1 1))
    (qe_191_0 : qw_191_0 = liftInteger qw_190_0)
    (qe_193_0 : qw_193_0 = multiplyMonomial qw_191_0 qw_192_0)
    (qe_194_0 : qw_194_0 = matrixMulScalarLeft qw_189_0 qw_193_0)
    (qe_195_0 : qw_195_0 = matrixMulScalarRight qw_188_0 qw_194_0)
    (qe_196_0 : qw_196_0 = matrixSub qw_184_0 qw_195_0)
    (qe_198_0 : qw_198_0 = matrixAdd qw_196_0 qw_197_0)
    (qe_199_0 : qw_199_0 = matrixMulScalarLeft qw_180_0 qw_177_0)
    (qe_200_0 : qw_200_0 = matrixMulScalarLeft qw_182_0 qw_179_0)
    (qe_201_0 : qw_201_0 = matrixAdd qw_199_0 qw_200_0)
    (qe_203_0 : qw_203_0 = matrixMulScalarLeft qw_202_0 qw_187_0)
    (qe_204_0 : qw_204_0 = matrixMulScalarRight qw_203_0 qw_194_0)
    (qe_205_0 : qw_205_0 = matrixSub qw_201_0 qw_204_0)
    (qe_207_0 : qw_207_0 = matrixAdd qw_205_0 qw_206_0)
    (qh218_1 : (∀ i : Fin 4, Stage_encrypt.parallel_parallel_generatedRoot_454_218 Backend.backend model params'' i0 i (qw_180_0, (qw_215_0, (qw_182_0, ()))) ((qw_218_0 i), ((qw_218_1 i), ()))))
    (qh214_6 : (∀ i : Fin 5, Stage_encrypt.parallel_parallel_generatedRoot_454_214 Backend.backend model params'' i0 i ((qw_208_0 i), ((qw_209_0 i), ((qw_210_0 i), ((qw_211_0 i), (qw_180_0, (qw_182_0, ((qw_212_0 i), ((qw_213_0 i), ())))))))) ((qw_214_0 i), ((qw_214_1 i), ((qw_214_2 i), ((qw_214_3 i), ((qw_214_4 i), ((qw_214_5 i), ((qw_214_6 i), ())))))))))
    (qh180_0 : centeredRebaseRuns qw_4_0 qw_180_0)
    (qh182_0 : centeredRebaseRuns qw_7_0 qw_182_0)
    (qh185_0 : centeredRebaseRuns qw_11_0 qw_185_0)
    (qh186_0 : gadgetMatrixRuns Backend.backend 32768 10 qw_186_0)
    (qh187_0 : centeredRebaseRuns qw_186_0 qw_187_0)
    (qh197_0 : gaussianSample ((4/1)) (26) qw_197_0)
    (qh202_0 : centeredRebaseRuns qw_29_0 qw_202_0)
    (qh206_0 : gaussianSample ((4/1)) (26) qw_206_0)
    {r sm : ErrorPoly N} (hr : polyNorm r ≤ 1) (hsm : polyNorm sm ≤ 1)
    (hpay : pw_9_0 = fun _ _ ↦ reducePoly Q0 N r) (qhm : qw_4_0 = fun _ _ ↦ reducePoly Q0 N sm)
    (qhp : qw_11_0 = fun _ _ ↦ reducePoly Q0 N r) (qm1 : qw_190_0 = 1) (qm2 : qw_192_0 = 0)
    {GQ6_12 : ExactMatrix Q6 N 1 12} {GQ5_10 : ExactMatrix Q5 N 1 10} {RQ6_10 : ExactMatrix Q6 N 1 10}
    {GQ0_8 : ExactMatrix Q0 N 1 8} {RQ5_8 : ExactMatrix Q5 N 1 8}
    {GQ1_6 : ExactMatrix (R i0) N 1 6} {RQ0_6 : ExactMatrix Q0 N 1 6}
    (hGQ6_12 : gadgetMatrixRuns Backend.backend 32768 12 GQ6_12)
    (hGQ5_10 : gadgetMatrixRuns Backend.backend 32768 10 GQ5_10) (hRQ6_10 : centeredRebaseRuns GQ5_10 RQ6_10)
    (hGQ0_8 : gadgetMatrixRuns Backend.backend 32768 8 GQ0_8) (hRQ5_8 : centeredRebaseRuns GQ0_8 RQ5_8)
    (hGQ1_6 : gadgetMatrixRuns Backend.backend 32768 6 GQ1_6) (hRQ0_6 : centeredRebaseRuns GQ1_6 RQ0_6)
    (hlc : ∀ k, qw_212_0 k * qw_212_0 k = 1)
    (hlr : ∀ k, qw_213_0 k * 2 + (1 - qw_212_0 k) * 8 = (labelExp k : Int) * 4)
    (hle : ∀ k, 0 ≤ qw_213_0 k ∧ qw_213_0 k ≤ 7)
    (hs01 : ∀ i, 0 ≤ pw_511_0 i ∧ pw_511_0 i ≤ 1) (hs01' : ∀ i, 0 ≤ pw_515_0 i ∧ pw_515_0 i ≤ 1)
    (he0 : ∀ i, pw_512_0 i = 0) (he0' : ∀ i, pw_516_0 i = 0)
    (hsum : ∀ o : Fin 160, ∑ k : Fin 5, pw_511_0 (cellIndex o k) = 1)
    (hsum' : ∀ o : Fin 160, ∑ k : Fin 5, pw_515_0 (cellIndex o k) = 1) :
    Rel sm ![![qw_176_0, qw_177_0], ![qw_178_0, qw_179_0]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair r j) • RQ6_10) ![qw_198_0, qw_207_0] 26 ∧
    (∀ i : Fin 4, Rel sm (lutA model qw_215_0 Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • RQ0_6) ![qw_218_0 i, qw_218_1 i] 26) ∧
    (∀ k : Fin 5, Rel sm ![![qw_214_0 k, qw_214_1 k], ![qw_214_2 k, qw_214_3 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp k) * unitPair sm j) • GQ6_12)
      ![qw_214_4 k, qw_214_5 k] 26) ∧
    (∃ eBm : Fin 800 → Fin 2 → ErrorMatrix N 1 8, (∀ i j, CoeffBound (eBm i j) 26) ∧ ∀ (i : Fin 800) j,
      reducePoly Q5 N r • rhsTop model pw_495_0 Q5 8 (smudge4Rhs0 i0) i j + ![pw_513_6 i, pw_513_7 i] j =
        reducePoly Q5 N (((p4 : Nat) : Int) * ((pw_511_0 i : Int) : ErrorPoly N) * unitPair r j) • RQ5_8 +
          reduceMatrix Q5 N 1 8 (eBm i j)) ∧
    (∃ eBm : Fin 800 → Fin 2 → ErrorMatrix N 1 6, (∀ i j, CoeffBound (eBm i j) 26) ∧ ∀ (i : Fin 800) j,
      reducePoly Q0 N r • rhsTop model pw_495_0 Q0 6 (smudge4Rhs1 i0) i j + ![pw_517_6 i, pw_517_7 i] j =
        reducePoly Q0 N ((((qd i0) : Nat) : Int) * ((pw_515_0 i : Int) : ErrorPoly N) * unitPair r j) • RQ0_6 +
          reduceMatrix Q0 N 1 6 (eBm i j)) ∧
    (∃ sel : Fin 160 → Fin 5, ∀ o k, pw_511_0 (cellIndex o k) = if k = sel o then 1 else 0) ∧
    (∃ sel : Fin 160 → Fin 5, ∀ o k, pw_515_0 (cellIndex o k) = if k = sel o then 1 else 0) ∧
    (∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) r) •
          ![![pw_506_0, pw_506_1], ![pw_506_2, pw_506_3]] 0 j i +
            ![![pw_506_0, pw_506_1], ![pw_506_2, pw_506_3]] 1 j i =
        (lutCoefficient (R i0) (inv4 i0) (fun k ↦ reducePoly (R i0) N (smudgeTable k))
          (sigma i).toNat * reducePoly (R i0) N ((1 : ErrorPoly N) * unitPair r j)) • GQ1_6 +
            reduceMatrix (R i0) N 1 6 (e j)) := by
  have hone : qw_29_0 = 1 := qe_29_0
  have h1 := encOne_198_0 (model := model) (e_7_0 := qe_7_0) (e_29_0 := qe_29_0) (e_181_0 := qe_181_0) (e_183_0 := qe_183_0) (e_184_0 := qe_184_0) (e_188_0 := qe_188_0) (e_189_0 := qe_189_0) (e_191_0 := qe_191_0) (e_193_0 := qe_193_0) (e_194_0 := qe_194_0) (e_195_0 := qe_195_0) (e_196_0 := qe_196_0) (e_198_0 := qe_198_0) (e_199_0 := qe_199_0) (e_200_0 := qe_200_0) (e_201_0 := qe_201_0) (e_203_0 := qe_203_0) (e_204_0 := qe_204_0) (e_205_0 := qe_205_0) (e_207_0 := qe_207_0) (h180_0 := qh180_0) (h182_0 := qh182_0) (h185_0 := qh185_0) (h186_0 := qh186_0) (h187_0 := qh187_0) (h197_0 := qh197_0) (h202_0 := qh202_0) (h206_0 := qh206_0) (hsm := hsm) (hsp := hr)
    (hsmIn := qhm) (hspIn := qhp) (hone := hone) (hg' := hGQ5_10) (hg := hRQ6_10)
  rw [qm1, qm2, message_one_zero] at h1
  have hsec : pw_504_0 = fun _ _ ↦ reducePoly Q6 N r :=
    rebase_small (q := Q0) (by inst_num) hr (by rw [← hpay]; exact ph504_0)
  have h1' : pw_505_0 = 1 := rebase_one (q := Q0) (by inst_num) (by rw [← pe_11_0]; exact ph505_0)
  have hsmQ6 : qw_180_0 = fun _ _ ↦ reducePoly Q6 N sm :=
    rebase_small (q := Q0) (by inst_num) hsm (by rw [← qhm]; exact qh180_0)
  have h343 : qw_182_0 = 1 := rebase_one (q := Q0) (by inst_num) (by rw [← qe_7_0]; exact qh182_0)
  have htab : (fun k ↦ reducePoly (R i0) N (smudgeTable k)) = fun k ↦ reducePoly (R i0) N
      (![((1 : Int) : ErrorPoly N), ((1 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N)] k) := by
    funext k; fin_cases k <;> rfl
  refine ⟨h1, fun i ↦ encHelper218 i (qh218_1 i) hsmQ6 h343 hGQ1_6 hRQ0_6,
    encLabelRel214 qh214_6 hsmQ6 h343 hGQ6_12 hlc hlr hle, ?_, ?_, ?_, ?_, fun i ↦ ?_⟩
  · have hc := fun i : Fin 800 ↦ preRhs513 (ph513_7 i) hsec h1' hGQ0_8 hRQ5_8
    choose eB heB hrel using hc
    refine ⟨eB, heB, fun i j ↦ ?_⟩
    have ht := preTop513 (ph513_7 i)
    have := hrel i j
    rw [ht, he0 i, message_zero] at this
    exact this
  · have hc := fun i : Fin 800 ↦ preRhs517 (ph517_7 i) hsec h1' hGQ1_6 hRQ0_6
    choose eB heB hrel using hc
    refine ⟨eB, heB, fun i j ↦ ?_⟩
    have ht := preTop517 (ph517_7 i)
    have := hrel i j
    rw [ht, he0' i, message_zero] at this
    exact this
  · choose sel hsel using fun o : Fin 160 ↦ onehot_of_sum (fun k ↦ hs01 (cellIndex o k)) (hsum o)
    exact ⟨sel, hsel⟩
  · choose sel hsel using fun o : Fin 160 ↦ onehot_of_sum (fun k ↦ hs01' (cellIndex o k)) (hsum' o)
    exact ⟨sel, hsel⟩
  · obtain ⟨e, he, hs⟩ := preSwitch506 i (ph506_3 i) hsec h1' hGQ1_6
    refine ⟨e, he, fun j ↦ ?_⟩
    rw [htab, show (1 : ErrorPoly N) = ((1 : Int) : ErrorPoly N) from Int.cast_one.symm]
    fin_cases j
    · cast_exact (hs 0)
    · cast_exact (hs 1)

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The inputs of the smudge chain of decoder `5`. -/
theorem smudgeInputs5 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params' : Stage_preprocess.Params} {params'' : Stage_encrypt.Params}
    {pw_9_0 : ExactMatrix Q0 8 1 1}
    {pw_539_0 : ByteArray}
    {pw_555_0 : Fin 800 → Int}
    {pw_556_0 : Fin 800 → Int}
    {pw_559_0 : Fin 800 → Int}
    {pw_560_0 : Fin 800 → Int}
    {pw_11_0 : ExactMatrix Q0 8 1 1}
    {pw_548_0 : ExactMatrix Q6 8 1 1}
    {pw_549_0 : ExactMatrix Q6 8 1 1}
    {pw_550_0 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_550_1 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_550_2 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_550_3 : (Fin 4 → ExactMatrix (R i0) 8 1 6)}
    {pw_551_0 : ExactMatrix Q6 8 1 10}
    {pw_552_0 : ExactMatrix Q6 8 1 10}
    {pw_553_0 : ExactMatrix Q6 8 1 10}
    {pw_554_0 : ExactMatrix Q6 8 1 10}
    {pw_557_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_557_1 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_557_2 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_557_3 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_557_4 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_557_5 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_557_6 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_557_7 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_558_0 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_558_1 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_558_2 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_558_3 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_561_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_561_1 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_561_2 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_561_3 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_561_4 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_561_5 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_561_6 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_561_7 : Fin 800 → ExactMatrix Q0 8 1 6}
    (pe_11_0 : pw_11_0 = (1 : ExactMatrix Q0 8 1 1))
    (ph548_0 : centeredRebaseRuns pw_9_0 pw_548_0)
    (ph549_0 : centeredRebaseRuns pw_11_0 pw_549_0)
    (ph557_7 : (∀ i : Fin 800, Stage_preprocess.parallel_parallel_generatedRoot_407_561 Backend.backend model params' i0 i (pw_539_0, (pw_551_0, (pw_552_0, (pw_553_0, (pw_554_0, (pw_548_0, ((pw_555_0 i), ((pw_556_0 i), (pw_549_0, ()))))))))) ((pw_557_0 i), ((pw_557_1 i), ((pw_557_2 i), ((pw_557_3 i), ((pw_557_4 i), ((pw_557_5 i), ((pw_557_6 i), ((pw_557_7 i), ()))))))))))
    (ph561_7 : (∀ i : Fin 800, Stage_preprocess.parallel_parallel_generatedRoot_407_565 Backend.backend model params' i0 i (pw_539_0, (pw_558_0, (pw_558_2, (pw_558_1, (pw_558_3, (pw_548_0, ((pw_559_0 i), ((pw_560_0 i), (pw_549_0, ()))))))))) ((pw_561_0 i), ((pw_561_1 i), ((pw_561_2 i), ((pw_561_3 i), ((pw_561_4 i), ((pw_561_5 i), ((pw_561_6 i), ((pw_561_7 i), ()))))))))))
    (ph550_3 : (∀ i : Fin 4, Stage_preprocess.parallel_parallel_generatedRoot_407_554 Backend.backend model params' i0 i (pw_539_0, (pw_548_0, (pw_549_0, ()))) ((pw_550_0 i), ((pw_550_1 i), ((pw_550_2 i), ((pw_550_3 i), ()))))))
    {qw_4_0 : ExactMatrix Q0 8 1 1}
    {qw_11_0 : ExactMatrix Q0 8 1 1}
    {qw_219_0 : ExactMatrix Q6 8 1 10}
    {qw_220_0 : ExactMatrix Q6 8 1 10}
    {qw_221_0 : ExactMatrix Q6 8 1 10}
    {qw_222_0 : ExactMatrix Q6 8 1 10}
    {qw_233_0 : Int}
    {qw_235_0 : Int}
    {qw_251_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_252_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_253_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_254_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_255_0 : Fin 5 → Int}
    {qw_256_0 : Fin 5 → Int}
    {qw_258_0 : ByteArray}
    {qw_7_0 : ExactMatrix Q0 8 1 1}
    {qw_29_0 : ExactMatrix Q0 8 1 1}
    {qw_223_0 : ExactMatrix Q6 8 1 1}
    {qw_224_0 : ExactMatrix Q6 8 1 10}
    {qw_225_0 : ExactMatrix Q6 8 1 1}
    {qw_226_0 : ExactMatrix Q6 8 1 10}
    {qw_227_0 : ExactMatrix Q6 8 1 10}
    {qw_228_0 : ExactMatrix Q6 8 1 1}
    {qw_229_0 : ExactMatrix Q5 8 1 10}
    {qw_230_0 : ExactMatrix Q6 8 1 10}
    {qw_231_0 : ExactMatrix Q6 8 1 10}
    {qw_232_0 : ExactMatrix Q6 8 1 1}
    {qw_234_0 : ExactMatrix Q6 8 1 1}
    {qw_236_0 : ExactMatrix Q6 8 1 1}
    {qw_237_0 : ExactMatrix Q6 8 1 1}
    {qw_238_0 : ExactMatrix Q6 8 1 10}
    {qw_239_0 : ExactMatrix Q6 8 1 10}
    {qw_240_0 : ExactMatrix Q6 8 1 10}
    {qw_241_0 : ExactMatrix Q6 8 1 10}
    {qw_242_0 : ExactMatrix Q6 8 1 10}
    {qw_243_0 : ExactMatrix Q6 8 1 10}
    {qw_244_0 : ExactMatrix Q6 8 1 10}
    {qw_245_0 : ExactMatrix Q6 8 1 1}
    {qw_246_0 : ExactMatrix Q6 8 1 10}
    {qw_247_0 : ExactMatrix Q6 8 1 10}
    {qw_248_0 : ExactMatrix Q6 8 1 10}
    {qw_249_0 : ExactMatrix Q6 8 1 10}
    {qw_250_0 : ExactMatrix Q6 8 1 10}
    {qw_257_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_257_1 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_257_2 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_257_3 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_257_4 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_257_5 : Fin 5 → ExactMatrix Q6 8 1 12}
    {qw_257_6 : Fin 5 → ExactMatrix Q6 8 1 1}
    {qw_261_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {qw_261_1 : Fin 4 → ExactMatrix Q0 8 1 6}
    (qe_7_0 : qw_7_0 = (1 : ExactMatrix Q0 8 1 1))
    (qe_29_0 : qw_29_0 = (1 : ExactMatrix Q0 8 1 1))
    (qe_224_0 : qw_224_0 = matrixMulScalarLeft qw_223_0 qw_219_0)
    (qe_226_0 : qw_226_0 = matrixMulScalarLeft qw_225_0 qw_221_0)
    (qe_227_0 : qw_227_0 = matrixAdd qw_224_0 qw_226_0)
    (qe_231_0 : qw_231_0 = matrixMulScalarLeft qw_228_0 qw_230_0)
    (qe_232_0 : qw_232_0 = (matrixPolynomial [(p5)] : ExactMatrix Q6 8 1 1))
    (qe_234_0 : qw_234_0 = liftInteger qw_233_0)
    (qe_236_0 : qw_236_0 = multiplyMonomial qw_234_0 qw_235_0)
    (qe_237_0 : qw_237_0 = matrixMulScalarLeft qw_232_0 qw_236_0)
    (qe_238_0 : qw_238_0 = matrixMulScalarRight qw_231_0 qw_237_0)
    (qe_239_0 : qw_239_0 = matrixSub qw_227_0 qw_238_0)
    (qe_241_0 : qw_241_0 = matrixAdd qw_239_0 qw_240_0)
    (qe_242_0 : qw_242_0 = matrixMulScalarLeft qw_223_0 qw_220_0)
    (qe_243_0 : qw_243_0 = matrixMulScalarLeft qw_225_0 qw_222_0)
    (qe_244_0 : qw_244_0 = matrixAdd qw_242_0 qw_243_0)
    (qe_246_0 : qw_246_0 = matrixMulScalarLeft qw_245_0 qw_230_0)
    (qe_247_0 : qw_247_0 = matrixMulScalarRight qw_246_0 qw_237_0)
    (qe_248_0 : qw_248_0 = matrixSub qw_244_0 qw_247_0)
    (qe_250_0 : qw_250_0 = matrixAdd qw_248_0 qw_249_0)
    (qh261_1 : (∀ i : Fin 4, Stage_encrypt.parallel_parallel_generatedRoot_454_261 Backend.backend model params'' i0 i (qw_223_0, (qw_258_0, (qw_225_0, ()))) ((qw_261_0 i), ((qw_261_1 i), ()))))
    (qh257_6 : (∀ i : Fin 5, Stage_encrypt.parallel_parallel_generatedRoot_454_257 Backend.backend model params'' i0 i ((qw_251_0 i), ((qw_252_0 i), ((qw_253_0 i), ((qw_254_0 i), (qw_223_0, (qw_225_0, ((qw_255_0 i), ((qw_256_0 i), ())))))))) ((qw_257_0 i), ((qw_257_1 i), ((qw_257_2 i), ((qw_257_3 i), ((qw_257_4 i), ((qw_257_5 i), ((qw_257_6 i), ())))))))))
    (qh223_0 : centeredRebaseRuns qw_4_0 qw_223_0)
    (qh225_0 : centeredRebaseRuns qw_7_0 qw_225_0)
    (qh228_0 : centeredRebaseRuns qw_11_0 qw_228_0)
    (qh229_0 : gadgetMatrixRuns Backend.backend 32768 10 qw_229_0)
    (qh230_0 : centeredRebaseRuns qw_229_0 qw_230_0)
    (qh240_0 : gaussianSample ((4/1)) (26) qw_240_0)
    (qh245_0 : centeredRebaseRuns qw_29_0 qw_245_0)
    (qh249_0 : gaussianSample ((4/1)) (26) qw_249_0)
    {r sm : ErrorPoly N} (hr : polyNorm r ≤ 1) (hsm : polyNorm sm ≤ 1)
    (hpay : pw_9_0 = fun _ _ ↦ reducePoly Q0 N r) (qhm : qw_4_0 = fun _ _ ↦ reducePoly Q0 N sm)
    (qhp : qw_11_0 = fun _ _ ↦ reducePoly Q0 N r) (qm1 : qw_233_0 = 1) (qm2 : qw_235_0 = 0)
    {GQ6_12 : ExactMatrix Q6 N 1 12} {GQ5_10 : ExactMatrix Q5 N 1 10} {RQ6_10 : ExactMatrix Q6 N 1 10}
    {GQ0_8 : ExactMatrix Q0 N 1 8} {RQ5_8 : ExactMatrix Q5 N 1 8}
    {GQ1_6 : ExactMatrix (R i0) N 1 6} {RQ0_6 : ExactMatrix Q0 N 1 6}
    (hGQ6_12 : gadgetMatrixRuns Backend.backend 32768 12 GQ6_12)
    (hGQ5_10 : gadgetMatrixRuns Backend.backend 32768 10 GQ5_10) (hRQ6_10 : centeredRebaseRuns GQ5_10 RQ6_10)
    (hGQ0_8 : gadgetMatrixRuns Backend.backend 32768 8 GQ0_8) (hRQ5_8 : centeredRebaseRuns GQ0_8 RQ5_8)
    (hGQ1_6 : gadgetMatrixRuns Backend.backend 32768 6 GQ1_6) (hRQ0_6 : centeredRebaseRuns GQ1_6 RQ0_6)
    (hlc : ∀ k, qw_255_0 k * qw_255_0 k = 1)
    (hlr : ∀ k, qw_256_0 k * 2 + (1 - qw_255_0 k) * 8 = (labelExp k : Int) * 4)
    (hle : ∀ k, 0 ≤ qw_256_0 k ∧ qw_256_0 k ≤ 7)
    (hs01 : ∀ i, 0 ≤ pw_555_0 i ∧ pw_555_0 i ≤ 1) (hs01' : ∀ i, 0 ≤ pw_559_0 i ∧ pw_559_0 i ≤ 1)
    (he0 : ∀ i, pw_556_0 i = 0) (he0' : ∀ i, pw_560_0 i = 0)
    (hsum : ∀ o : Fin 160, ∑ k : Fin 5, pw_555_0 (cellIndex o k) = 1)
    (hsum' : ∀ o : Fin 160, ∑ k : Fin 5, pw_559_0 (cellIndex o k) = 1) :
    Rel sm ![![qw_219_0, qw_220_0], ![qw_221_0, qw_222_0]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair r j) • RQ6_10) ![qw_241_0, qw_250_0] 26 ∧
    (∀ i : Fin 4, Rel sm (lutA model qw_258_0 Q0 6 2 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • RQ0_6) ![qw_261_0 i, qw_261_1 i] 26) ∧
    (∀ k : Fin 5, Rel sm ![![qw_257_0 k, qw_257_1 k], ![qw_257_2 k, qw_257_3 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp k) * unitPair sm j) • GQ6_12)
      ![qw_257_4 k, qw_257_5 k] 26) ∧
    (∃ eBm : Fin 800 → Fin 2 → ErrorMatrix N 1 8, (∀ i j, CoeffBound (eBm i j) 26) ∧ ∀ (i : Fin 800) j,
      reducePoly Q5 N r • rhsTop model pw_539_0 Q5 8 (smudge5Rhs0 i0) i j + ![pw_557_6 i, pw_557_7 i] j =
        reducePoly Q5 N (((p4 : Nat) : Int) * ((pw_555_0 i : Int) : ErrorPoly N) * unitPair r j) • RQ5_8 +
          reduceMatrix Q5 N 1 8 (eBm i j)) ∧
    (∃ eBm : Fin 800 → Fin 2 → ErrorMatrix N 1 6, (∀ i j, CoeffBound (eBm i j) 26) ∧ ∀ (i : Fin 800) j,
      reducePoly Q0 N r • rhsTop model pw_539_0 Q0 6 (smudge5Rhs1 i0) i j + ![pw_561_6 i, pw_561_7 i] j =
        reducePoly Q0 N ((((qd i0) : Nat) : Int) * ((pw_559_0 i : Int) : ErrorPoly N) * unitPair r j) • RQ0_6 +
          reduceMatrix Q0 N 1 6 (eBm i j)) ∧
    (∃ sel : Fin 160 → Fin 5, ∀ o k, pw_555_0 (cellIndex o k) = if k = sel o then 1 else 0) ∧
    (∃ sel : Fin 160 → Fin 5, ∀ o k, pw_559_0 (cellIndex o k) = if k = sel o then 1 else 0) ∧
    (∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) r) •
          ![![pw_550_0, pw_550_1], ![pw_550_2, pw_550_3]] 0 j i +
            ![![pw_550_0, pw_550_1], ![pw_550_2, pw_550_3]] 1 j i =
        (lutCoefficient (R i0) (inv4 i0) (fun k ↦ reducePoly (R i0) N (smudgeTable k))
          (sigma i).toNat * reducePoly (R i0) N ((1 : ErrorPoly N) * unitPair r j)) • GQ1_6 +
            reduceMatrix (R i0) N 1 6 (e j)) := by
  have hone : qw_29_0 = 1 := qe_29_0
  have h1 := encOne_241_0 (model := model) (e_7_0 := qe_7_0) (e_29_0 := qe_29_0) (e_224_0 := qe_224_0) (e_226_0 := qe_226_0) (e_227_0 := qe_227_0) (e_231_0 := qe_231_0) (e_232_0 := qe_232_0) (e_234_0 := qe_234_0) (e_236_0 := qe_236_0) (e_237_0 := qe_237_0) (e_238_0 := qe_238_0) (e_239_0 := qe_239_0) (e_241_0 := qe_241_0) (e_242_0 := qe_242_0) (e_243_0 := qe_243_0) (e_244_0 := qe_244_0) (e_246_0 := qe_246_0) (e_247_0 := qe_247_0) (e_248_0 := qe_248_0) (e_250_0 := qe_250_0) (h223_0 := qh223_0) (h225_0 := qh225_0) (h228_0 := qh228_0) (h229_0 := qh229_0) (h230_0 := qh230_0) (h240_0 := qh240_0) (h245_0 := qh245_0) (h249_0 := qh249_0) (hsm := hsm) (hsp := hr)
    (hsmIn := qhm) (hspIn := qhp) (hone := hone) (hg' := hGQ5_10) (hg := hRQ6_10)
  rw [qm1, qm2, message_one_zero] at h1
  have hsec : pw_548_0 = fun _ _ ↦ reducePoly Q6 N r :=
    rebase_small (q := Q0) (by inst_num) hr (by rw [← hpay]; exact ph548_0)
  have h1' : pw_549_0 = 1 := rebase_one (q := Q0) (by inst_num) (by rw [← pe_11_0]; exact ph549_0)
  have hsmQ6 : qw_223_0 = fun _ _ ↦ reducePoly Q6 N sm :=
    rebase_small (q := Q0) (by inst_num) hsm (by rw [← qhm]; exact qh223_0)
  have h343 : qw_225_0 = 1 := rebase_one (q := Q0) (by inst_num) (by rw [← qe_7_0]; exact qh225_0)
  have htab : (fun k ↦ reducePoly (R i0) N (smudgeTable k)) = fun k ↦ reducePoly (R i0) N
      (![((1 : Int) : ErrorPoly N), ((1 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N)] k) := by
    funext k; fin_cases k <;> rfl
  refine ⟨h1, fun i ↦ encHelper261 i (qh261_1 i) hsmQ6 h343 hGQ1_6 hRQ0_6,
    encLabelRel257 qh257_6 hsmQ6 h343 hGQ6_12 hlc hlr hle, ?_, ?_, ?_, ?_, fun i ↦ ?_⟩
  · have hc := fun i : Fin 800 ↦ preRhs557 (ph557_7 i) hsec h1' hGQ0_8 hRQ5_8
    choose eB heB hrel using hc
    refine ⟨eB, heB, fun i j ↦ ?_⟩
    have ht := preTop557 (ph557_7 i)
    have := hrel i j
    rw [ht, he0 i, message_zero] at this
    exact this
  · have hc := fun i : Fin 800 ↦ preRhs561 (ph561_7 i) hsec h1' hGQ1_6 hRQ0_6
    choose eB heB hrel using hc
    refine ⟨eB, heB, fun i j ↦ ?_⟩
    have ht := preTop561 (ph561_7 i)
    have := hrel i j
    rw [ht, he0' i, message_zero] at this
    exact this
  · choose sel hsel using fun o : Fin 160 ↦ onehot_of_sum (fun k ↦ hs01 (cellIndex o k)) (hsum o)
    exact ⟨sel, hsel⟩
  · choose sel hsel using fun o : Fin 160 ↦ onehot_of_sum (fun k ↦ hs01' (cellIndex o k)) (hsum' o)
    exact ⟨sel, hsel⟩
  · obtain ⟨e, he, hs⟩ := preSwitch550 i (ph550_3 i) hsec h1' hGQ1_6
    refine ⟨e, he, fun j ↦ ?_⟩
    rw [htab, show (1 : ErrorPoly N) = ((1 : Int) : ErrorPoly N) from Int.cast_one.symm]
    fin_cases j
    · cast_exact (hs 0)
    · cast_exact (hs 1)

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The inputs of the source LUT. -/
theorem sourceInputs {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params' : Stage_preprocess.Params} {params'' : Stage_encrypt.Params}
    {pw_9_0 : ExactMatrix Q0 8 1 1}
    {pw_168_0 : ByteArray}
    {pw_11_0 : ExactMatrix Q0 8 1 1}
    {pw_182_0 : (Fin 2 → ExactMatrix (R i0) 8 1 4)}
    {pw_182_1 : (Fin 2 → ExactMatrix (R i0) 8 1 4)}
    {pw_182_2 : (Fin 2 → ExactMatrix (R i0) 8 1 4)}
    {pw_182_3 : (Fin 2 → ExactMatrix (R i0) 8 1 4)}
    (pe_11_0 : pw_11_0 = (1 : ExactMatrix Q0 8 1 1))
    (ph182_3 : (∀ i : Fin 2, Stage_preprocess.parallel_parallel_generatedRoot_407_183 Backend.backend model params' i0 i (pw_168_0, (pw_9_0, (pw_11_0, ()))) ((pw_182_0 i), ((pw_182_1 i), ((pw_182_2 i), ((pw_182_3 i), ()))))))
    {qw_4_0 : ExactMatrix Q0 8 1 1}
    {qw_11_0 : ExactMatrix Q0 8 1 1}
    {qw_262_0 : ByteArray}
    {qw_272_0 : Int}
    {qw_274_0 : Int}
    {qw_7_0 : ExactMatrix Q0 8 1 1}
    {qw_29_0 : ExactMatrix Q0 8 1 1}
    {qw_263_0 : ExactMatrix Q0 8 1 6}
    {qw_264_0 : ExactMatrix Q0 8 1 6}
    {qw_265_0 : ExactMatrix Q0 8 1 6}
    {qw_266_0 : ExactMatrix Q0 8 1 6}
    {qw_267_0 : ExactMatrix Q0 8 1 6}
    {qw_268_0 : ExactMatrix (R i0) 8 1 6}
    {qw_269_0 : ExactMatrix Q0 8 1 6}
    {qw_270_0 : ExactMatrix Q0 8 1 6}
    {qw_271_0 : ExactMatrix Q0 8 1 1}
    {qw_273_0 : ExactMatrix Q0 8 1 1}
    {qw_275_0 : ExactMatrix Q0 8 1 1}
    {qw_276_0 : ExactMatrix Q0 8 1 1}
    {qw_277_0 : ExactMatrix Q0 8 1 6}
    {qw_278_0 : ExactMatrix Q0 8 1 6}
    {qw_279_0 : ExactMatrix Q0 8 1 6}
    {qw_280_0 : ExactMatrix Q0 8 1 6}
    {qw_281_0 : ExactMatrix Q0 8 1 6}
    {qw_282_0 : ExactMatrix Q0 8 1 6}
    {qw_283_0 : ExactMatrix Q0 8 1 6}
    {qw_284_0 : ExactMatrix Q0 8 1 6}
    {qw_285_0 : ExactMatrix Q0 8 1 6}
    {qw_286_0 : ExactMatrix Q0 8 1 6}
    {qw_287_0 : ExactMatrix Q0 8 1 6}
    {qw_288_0 : ExactMatrix Q0 8 1 6}
    {qw_289_0 : ExactMatrix Q0 8 1 6}
    {qw_290_0 : ExactMatrix Q0 8 1 6}
    {qw_291_0 : Fin 2 → ExactMatrix Q0 8 1 6}
    {qw_291_1 : Fin 2 → ExactMatrix Q0 8 1 6}
    (qe_7_0 : qw_7_0 = (1 : ExactMatrix Q0 8 1 1))
    (qe_29_0 : qw_29_0 = (1 : ExactMatrix Q0 8 1 1))
    (qe_264_0 : qw_264_0 = matrixMulScalarLeft qw_4_0 qw_263_0)
    (qe_266_0 : qw_266_0 = matrixMulScalarLeft qw_7_0 qw_265_0)
    (qe_267_0 : qw_267_0 = matrixAdd qw_264_0 qw_266_0)
    (qe_270_0 : qw_270_0 = matrixMulScalarLeft qw_11_0 qw_269_0)
    (qe_271_0 : qw_271_0 = (matrixPolynomial [((qdZ (i0 : Int)))] : ExactMatrix Q0 8 1 1))
    (qe_273_0 : qw_273_0 = liftInteger qw_272_0)
    (qe_275_0 : qw_275_0 = multiplyMonomial qw_273_0 qw_274_0)
    (qe_276_0 : qw_276_0 = matrixMulScalarLeft qw_271_0 qw_275_0)
    (qe_277_0 : qw_277_0 = matrixMulScalarRight qw_270_0 qw_276_0)
    (qe_278_0 : qw_278_0 = matrixSub qw_267_0 qw_277_0)
    (qe_280_0 : qw_280_0 = matrixAdd qw_278_0 qw_279_0)
    (qe_282_0 : qw_282_0 = matrixMulScalarLeft qw_4_0 qw_281_0)
    (qe_284_0 : qw_284_0 = matrixMulScalarLeft qw_7_0 qw_283_0)
    (qe_285_0 : qw_285_0 = matrixAdd qw_282_0 qw_284_0)
    (qe_286_0 : qw_286_0 = matrixMulScalarLeft qw_29_0 qw_269_0)
    (qe_287_0 : qw_287_0 = matrixMulScalarRight qw_286_0 qw_276_0)
    (qe_288_0 : qw_288_0 = matrixSub qw_285_0 qw_287_0)
    (qe_290_0 : qw_290_0 = matrixAdd qw_288_0 qw_289_0)
    (qh291_1 : (∀ i : Fin 2, Stage_encrypt.parallel_parallel_generatedRoot_454_291 Backend.backend model params'' i0 i (qw_4_0, (qw_262_0, (qw_7_0, ()))) ((qw_291_0 i), ((qw_291_1 i), ()))))
    (qh268_0 : gadgetMatrixRuns Backend.backend 32768 6 qw_268_0)
    (qh269_0 : centeredRebaseRuns qw_268_0 qw_269_0)
    (qh279_0 : gaussianSample ((4/1)) (26) qw_279_0)
    (qh289_0 : gaussianSample ((4/1)) (26) qw_289_0)
    {r sm : ErrorPoly N} (hr : polyNorm r ≤ 1) (hsm : polyNorm sm ≤ 1)
    (hpay : pw_9_0 = fun _ _ ↦ reducePoly Q0 N r) (qhm : qw_4_0 = fun _ _ ↦ reducePoly Q0 N sm)
    (qhp : qw_11_0 = fun _ _ ↦ reducePoly Q0 N r) (q272 : qw_272_0 = 1) (q274 : 0 ≤ qw_274_0 ∧ qw_274_0 ≤ 1)
    {GQ1_6 : ExactMatrix (R i0) N 1 6} {RQ0_6 : ExactMatrix Q0 N 1 6}
    {GM01_4 : ExactMatrix (Mx i0) N 1 4} {RQ1_4 : ExactMatrix (R i0) N 1 4}
    (hGQ1_6 : gadgetMatrixRuns Backend.backend 32768 6 GQ1_6) (hRQ0_6 : centeredRebaseRuns GQ1_6 RQ0_6)
    (hGM01_4 : gadgetMatrixRuns Backend.backend 32768 4 GM01_4) (hRQ1_4 : centeredRebaseRuns GM01_4 RQ1_4) :
    Rel sm ![![qw_263_0, qw_281_0], ![qw_265_0, qw_283_0]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * Xe (n := N) qw_274_0.toNat * unitPair r j) • RQ0_6)
      ![qw_280_0, qw_290_0] 26 ∧
    (∀ i : Fin 2, Rel sm (lutA model qw_262_0 Q0 6 0 (sigma2 i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma2 i).toNat (sigma2_odd i) sm) j) • RQ0_6) ![qw_291_0 i, qw_291_1 i] 26) ∧
    (∀ i : Fin 2, ∃ e : Fin 2 → ErrorMatrix N 1 4, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma2 i).toNat (sigma2_odd i) r) •
          ![![pw_182_0, pw_182_1], ![pw_182_2, pw_182_3]] 0 j i +
            ![![pw_182_0, pw_182_1], ![pw_182_2, pw_182_3]] 1 j i =
        (lutCoefficient2 (R i0) (half i0) (fun k ↦ reducePoly (R i0) N (sourceTable k))
          (sigma2 i).toNat * reducePoly (R i0) N (((((pp i0) : Nat) : Int) : ErrorPoly N) * unitPair r j)) • RQ1_4 +
            reduceMatrix (R i0) N 1 4 (e j)) := by
  have h1 := encOne_280_0 (model := model) (e_7_0 := qe_7_0) (e_29_0 := qe_29_0) (e_264_0 := qe_264_0) (e_266_0 := qe_266_0) (e_267_0 := qe_267_0) (e_270_0 := qe_270_0) (e_271_0 := qe_271_0) (e_273_0 := qe_273_0) (e_275_0 := qe_275_0) (e_276_0 := qe_276_0) (e_277_0 := qe_277_0) (e_278_0 := qe_278_0) (e_280_0 := qe_280_0) (e_282_0 := qe_282_0) (e_284_0 := qe_284_0) (e_285_0 := qe_285_0) (e_286_0 := qe_286_0) (e_287_0 := qe_287_0) (e_288_0 := qe_288_0) (e_290_0 := qe_290_0) (h268_0 := qh268_0) (h269_0 := qh269_0) (h279_0 := qh279_0) (h289_0 := qh289_0) (hsm := hsm) (hsp := hr)
    (hsmIn := qhm) (hspIn := qhp) (hg' := hGQ1_6) (hg := hRQ0_6)
  rw [q272, message_one_exp q274.1 (by omega)] at h1
  have htab : (fun k ↦ reducePoly (R i0) N (sourceTable k)) =
      fun k ↦ reducePoly (R i0) N (![Xe (n := N) 1, Xe (n := N) 0] k) := by
    funext k; fin_cases k <;> simp [sourceTable, Xe]
  refine ⟨h1, fun i ↦ encHelper291 i (qh291_1 i) qhm qe_7_0 hGQ1_6 hRQ0_6, fun i ↦ ?_⟩
  obtain ⟨e, he, hs⟩ := preSwitch182 i (ph182_3 i) hpay pe_11_0 hGM01_4 hRQ1_4
  refine ⟨e, he, fun j ↦ ?_⟩
  rw [htab]
  fin_cases j
  · cast_exact (hs 0)
  · cast_exact (hs 1)

end RefreshTopProof.Part_I
