import Mxx.Certificate.Rules.MatrixRules

namespace Mxx.Certificate

/-- An exact sampler relation can rewrite an executable product in the negacyclic quotient.
The premise is deliberately `MatrixModEq`: preimage and gadget-decomposition contracts only
establish equality in `R_q`, never equality of the stored integer representatives. -/
theorem exactRelationProduct_matrixValue
    (q ringDimension rows inner columns : Nat)
    [NeZero q] [NeZero ringDimension]
    (basis subject target : Mxx.Matrix)
    (basisLayout : Mxx.Toolkit.MatrixLayout basis q ringDimension rows inner)
    (subjectLayout : Mxx.Toolkit.MatrixLayout subject q ringDimension inner columns)
    (targetLayout : Mxx.Toolkit.MatrixLayout target q ringDimension rows columns)
    (relation : Mxx.MatrixModEq (Mxx.matrixMul basis subject) target) :
    Mxx.Toolkit.matrixValue q ringDimension rows columns
        (Mxx.matrixMultiply basis subject) =
      Mxx.Toolkit.matrixValue q ringDimension rows columns target := by
  rw [Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension rows inner columns
    basis subject basisLayout subjectLayout]
  rw [← Mxx.Toolkit.matrixValue_mul q ringDimension rows inner columns basis subject
    ⟨basisLayout.modulus, basisLayout.ringDimension, basisLayout.rows, basisLayout.columns⟩
    ⟨subjectLayout.modulus, subjectLayout.ringDimension, subjectLayout.rows,
      subjectLayout.columns⟩]
  exact Mxx.Toolkit.matrixValue_eq_of_modEq q ringDimension rows columns
    (Mxx.matrixMul basis subject) target
    (Mxx.Toolkit.matrixMul_layout basis subject basisLayout subjectLayout) targetLayout relation

/-- The one-term affine rewrite through an exact relation is valid in `R_q`:
`(s * B + e) * K = s * target + e * K` whenever `B * K = target` in `R_q`.
No equality of stored integer representatives is assumed. -/
theorem affineExactRelationProduct_matrixValue
    (q ringDimension outputRows basisRows inner columns : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (coefficient basis noise subject target : Mxx.Matrix)
    (coefficientLayout : Mxx.Toolkit.MatrixLayout coefficient q ringDimension
      outputRows basisRows)
    (basisLayout : Mxx.Toolkit.MatrixLayout basis q ringDimension basisRows inner)
    (noiseLayout : Mxx.Toolkit.MatrixLayout noise q ringDimension outputRows inner)
    (subjectLayout : Mxx.Toolkit.MatrixLayout subject q ringDimension inner columns)
    (targetLayout : Mxx.Toolkit.MatrixLayout target q ringDimension basisRows columns)
    (relation : Mxx.MatrixModEq (Mxx.matrixMul basis subject) target) :
    Mxx.Toolkit.matrixValue q ringDimension outputRows columns
        (Mxx.matrixMultiply
          (Mxx.matrixAdd (Mxx.matrixMultiply coefficient basis) noise) subject) =
      Mxx.Toolkit.matrixValue q ringDimension outputRows columns
        (Mxx.matrixAdd (Mxx.matrixMultiply coefficient target)
          (Mxx.matrixMultiply noise subject)) := by
  have coefficientBasisLayout := Mxx.Toolkit.matrixMultiply_layout coefficient basis
    coefficientLayout basisLayout
  have leftLayout := Mxx.Toolkit.matrixAdd_layout
    (Mxx.matrixMultiply coefficient basis) noise coefficientBasisLayout noiseLayout
  have coefficientTargetLayout := Mxx.Toolkit.matrixMultiply_layout coefficient target
    coefficientLayout targetLayout
  have noiseSubjectLayout := Mxx.Toolkit.matrixMultiply_layout noise subject
    noiseLayout subjectLayout
  rw [Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension outputRows inner columns
      (Mxx.matrixAdd (Mxx.matrixMultiply coefficient basis) noise) subject
      leftLayout subjectLayout,
    Mxx.Toolkit.matrixValue_add q ringDimension outputRows inner
      (Mxx.matrixMultiply coefficient basis) noise
      ⟨coefficientBasisLayout.modulus, coefficientBasisLayout.ringDimension,
        coefficientBasisLayout.rows, coefficientBasisLayout.columns⟩
      ⟨noiseLayout.modulus, noiseLayout.ringDimension, noiseLayout.rows, noiseLayout.columns⟩,
    Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension outputRows basisRows inner
      coefficient basis coefficientLayout basisLayout,
    Mxx.Toolkit.matrixValue_add q ringDimension outputRows columns
      (Mxx.matrixMultiply coefficient target) (Mxx.matrixMultiply noise subject)
      ⟨coefficientTargetLayout.modulus, coefficientTargetLayout.ringDimension,
        coefficientTargetLayout.rows, coefficientTargetLayout.columns⟩
      ⟨noiseSubjectLayout.modulus, noiseSubjectLayout.ringDimension,
        noiseSubjectLayout.rows, noiseSubjectLayout.columns⟩,
    Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension outputRows basisRows columns
      coefficient target coefficientLayout targetLayout,
    Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension outputRows inner columns
      noise subject noiseLayout subjectLayout]
  have basisSubject := exactRelationProduct_matrixValue q ringDimension basisRows inner columns
    basis subject target basisLayout subjectLayout targetLayout relation
  rw [Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension basisRows inner columns
    basis subject basisLayout subjectLayout] at basisSubject
  rw [Matrix.add_mul, Matrix.mul_assoc, basisSubject]

/-- If the relation target itself has affine form `c * P + E`, the elaborator may expand
`s * target` without introducing a fold/derived atom.  The resulting signal is `(s * c) * P`
and the resulting noise is `s * E + e * K`, all in the exact quotient `R_q`. -/
theorem affineRelationProduct_matrixValue
    (q ringDimension outputRows basisRows inner targetInner columns : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (signalCoefficient basis inputNoise subject target targetCoefficient targetBasis
      targetNoise : Mxx.Matrix)
    (signalCoefficientLayout : Mxx.Toolkit.MatrixLayout signalCoefficient q ringDimension
      outputRows basisRows)
    (basisLayout : Mxx.Toolkit.MatrixLayout basis q ringDimension basisRows inner)
    (inputNoiseLayout : Mxx.Toolkit.MatrixLayout inputNoise q ringDimension outputRows inner)
    (subjectLayout : Mxx.Toolkit.MatrixLayout subject q ringDimension inner columns)
    (targetLayout : Mxx.Toolkit.MatrixLayout target q ringDimension basisRows columns)
    (targetCoefficientLayout : Mxx.Toolkit.MatrixLayout targetCoefficient q ringDimension
      basisRows targetInner)
    (targetBasisLayout : Mxx.Toolkit.MatrixLayout targetBasis q ringDimension targetInner columns)
    (targetNoiseLayout : Mxx.Toolkit.MatrixLayout targetNoise q ringDimension basisRows columns)
    (basisRelation : Mxx.MatrixModEq (Mxx.matrixMul basis subject) target)
    (targetRelation : Mxx.MatrixModEq target
      (Mxx.matrixAdd (Mxx.matrixMultiply targetCoefficient targetBasis) targetNoise)) :
    Mxx.Toolkit.matrixValue q ringDimension outputRows columns
        (Mxx.matrixMultiply
          (Mxx.matrixAdd (Mxx.matrixMultiply signalCoefficient basis) inputNoise) subject) =
      Mxx.Toolkit.matrixValue q ringDimension outputRows columns
        (Mxx.matrixAdd
          (Mxx.matrixMultiply
            (Mxx.matrixMultiply signalCoefficient targetCoefficient) targetBasis)
          (Mxx.matrixAdd (Mxx.matrixMultiply signalCoefficient targetNoise)
            (Mxx.matrixMultiply inputNoise subject))) := by
  have targetProductLayout := Mxx.Toolkit.matrixMultiply_layout targetCoefficient targetBasis
    targetCoefficientLayout targetBasisLayout
  have targetReconstructionLayout := Mxx.Toolkit.matrixAdd_layout
    (Mxx.matrixMultiply targetCoefficient targetBasis) targetNoise
    targetProductLayout targetNoiseLayout
  have signalTargetLayout := Mxx.Toolkit.matrixMultiply_layout signalCoefficient target
    signalCoefficientLayout targetLayout
  have signalCoefficientProductLayout := Mxx.Toolkit.matrixMultiply_layout
    signalCoefficient targetCoefficient signalCoefficientLayout targetCoefficientLayout
  have rewrittenSignalLayout := Mxx.Toolkit.matrixMultiply_layout
    (Mxx.matrixMultiply signalCoefficient targetCoefficient) targetBasis
    signalCoefficientProductLayout targetBasisLayout
  have signalTargetNoiseLayout := Mxx.Toolkit.matrixMultiply_layout signalCoefficient targetNoise
    signalCoefficientLayout targetNoiseLayout
  have inputNoiseSubjectLayout := Mxx.Toolkit.matrixMultiply_layout inputNoise subject
    inputNoiseLayout subjectLayout
  have rewrittenNoiseLayout := Mxx.Toolkit.matrixAdd_layout
    (Mxx.matrixMultiply signalCoefficient targetNoise)
    (Mxx.matrixMultiply inputNoise subject) signalTargetNoiseLayout inputNoiseSubjectLayout
  rw [affineExactRelationProduct_matrixValue q ringDimension outputRows basisRows inner columns
    signalCoefficient basis inputNoise subject target signalCoefficientLayout basisLayout
    inputNoiseLayout subjectLayout targetLayout basisRelation,
    Mxx.Toolkit.matrixValue_add q ringDimension outputRows columns
      (Mxx.matrixMultiply signalCoefficient target)
      (Mxx.matrixMultiply inputNoise subject)
      ⟨signalTargetLayout.modulus, signalTargetLayout.ringDimension,
        signalTargetLayout.rows, signalTargetLayout.columns⟩
      ⟨inputNoiseSubjectLayout.modulus, inputNoiseSubjectLayout.ringDimension,
        inputNoiseSubjectLayout.rows, inputNoiseSubjectLayout.columns⟩,
    Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension outputRows basisRows columns
      signalCoefficient target signalCoefficientLayout targetLayout]
  have targetValue := Mxx.Toolkit.matrixValue_eq_of_modEq q ringDimension basisRows columns
    target (Mxx.matrixAdd (Mxx.matrixMultiply targetCoefficient targetBasis) targetNoise)
    targetLayout targetReconstructionLayout targetRelation
  rw [targetValue,
    Mxx.Toolkit.matrixValue_add q ringDimension basisRows columns
      (Mxx.matrixMultiply targetCoefficient targetBasis) targetNoise
      ⟨targetProductLayout.modulus, targetProductLayout.ringDimension,
        targetProductLayout.rows, targetProductLayout.columns⟩
      ⟨targetNoiseLayout.modulus, targetNoiseLayout.ringDimension,
        targetNoiseLayout.rows, targetNoiseLayout.columns⟩,
    Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension basisRows targetInner columns
      targetCoefficient targetBasis targetCoefficientLayout targetBasisLayout,
    Mxx.Toolkit.matrixValue_add q ringDimension outputRows columns
      (Mxx.matrixMultiply
        (Mxx.matrixMultiply signalCoefficient targetCoefficient) targetBasis)
      (Mxx.matrixAdd (Mxx.matrixMultiply signalCoefficient targetNoise)
        (Mxx.matrixMultiply inputNoise subject))
      ⟨rewrittenSignalLayout.modulus, rewrittenSignalLayout.ringDimension,
        rewrittenSignalLayout.rows, rewrittenSignalLayout.columns⟩
      ⟨rewrittenNoiseLayout.modulus, rewrittenNoiseLayout.ringDimension,
        rewrittenNoiseLayout.rows, rewrittenNoiseLayout.columns⟩,
    Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension outputRows targetInner columns
      (Mxx.matrixMultiply signalCoefficient targetCoefficient) targetBasis
      signalCoefficientProductLayout targetBasisLayout,
    Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension outputRows basisRows targetInner
      signalCoefficient targetCoefficient signalCoefficientLayout targetCoefficientLayout,
    Mxx.Toolkit.matrixValue_add q ringDimension outputRows columns
      (Mxx.matrixMultiply signalCoefficient targetNoise)
      (Mxx.matrixMultiply inputNoise subject)
      ⟨signalTargetNoiseLayout.modulus, signalTargetNoiseLayout.ringDimension,
        signalTargetNoiseLayout.rows, signalTargetNoiseLayout.columns⟩
      ⟨inputNoiseSubjectLayout.modulus, inputNoiseSubjectLayout.ringDimension,
        inputNoiseSubjectLayout.rows, inputNoiseSubjectLayout.columns⟩,
    Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension outputRows basisRows columns
      signalCoefficient targetNoise signalCoefficientLayout targetNoiseLayout,
    Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension outputRows inner columns
      inputNoise subject inputNoiseLayout subjectLayout]
  rw [Matrix.mul_add, Matrix.mul_assoc, add_assoc]

/-- The affine preimage rewrite as the executable `MatrixModEq` relation consumed by symbolic
facts.  This is the same quotient-ring theorem as `affineRelationProduct_matrixValue`; complete
layouts merely convert its result back to canonical coefficient congruence. -/
theorem affineRelationProduct_modEq
    (q ringDimension outputRows basisRows inner targetInner columns : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (signalCoefficient basis inputNoise subject target targetCoefficient targetBasis
      targetNoise : Mxx.Matrix)
    (signalCoefficientLayout : Mxx.Toolkit.MatrixLayout signalCoefficient q ringDimension
      outputRows basisRows)
    (basisLayout : Mxx.Toolkit.MatrixLayout basis q ringDimension basisRows inner)
    (inputNoiseLayout : Mxx.Toolkit.MatrixLayout inputNoise q ringDimension outputRows inner)
    (subjectLayout : Mxx.Toolkit.MatrixLayout subject q ringDimension inner columns)
    (targetLayout : Mxx.Toolkit.MatrixLayout target q ringDimension basisRows columns)
    (targetCoefficientLayout : Mxx.Toolkit.MatrixLayout targetCoefficient q ringDimension
      basisRows targetInner)
    (targetBasisLayout : Mxx.Toolkit.MatrixLayout targetBasis q ringDimension targetInner columns)
    (targetNoiseLayout : Mxx.Toolkit.MatrixLayout targetNoise q ringDimension basisRows columns)
    (basisRelation : Mxx.MatrixModEq (Mxx.matrixMul basis subject) target)
    (targetRelation : Mxx.MatrixModEq target
      (Mxx.matrixAdd (Mxx.matrixMultiply targetCoefficient targetBasis) targetNoise)) :
    Mxx.MatrixModEq
      (Mxx.matrixMultiply
        (Mxx.matrixAdd (Mxx.matrixMultiply signalCoefficient basis) inputNoise) subject)
      (Mxx.matrixAdd
        (Mxx.matrixMultiply
          (Mxx.matrixMultiply signalCoefficient targetCoefficient) targetBasis)
        (Mxx.matrixAdd (Mxx.matrixMultiply signalCoefficient targetNoise)
          (Mxx.matrixMultiply inputNoise subject))) := by
  have coefficientBasisLayout := Mxx.Toolkit.matrixMultiply_layout signalCoefficient basis
    signalCoefficientLayout basisLayout
  have affineInputLayout := Mxx.Toolkit.matrixAdd_layout
    (Mxx.matrixMultiply signalCoefficient basis) inputNoise
    coefficientBasisLayout inputNoiseLayout
  have leftLayout := Mxx.Toolkit.matrixMultiply_layout
    (Mxx.matrixAdd (Mxx.matrixMultiply signalCoefficient basis) inputNoise) subject
    affineInputLayout subjectLayout
  have coefficientProductLayout := Mxx.Toolkit.matrixMultiply_layout
    signalCoefficient targetCoefficient signalCoefficientLayout targetCoefficientLayout
  have signalLayout := Mxx.Toolkit.matrixMultiply_layout
    (Mxx.matrixMultiply signalCoefficient targetCoefficient) targetBasis
    coefficientProductLayout targetBasisLayout
  have signalNoiseLayout := Mxx.Toolkit.matrixMultiply_layout signalCoefficient targetNoise
    signalCoefficientLayout targetNoiseLayout
  have inputNoiseSubjectLayout := Mxx.Toolkit.matrixMultiply_layout inputNoise subject
    inputNoiseLayout subjectLayout
  have noiseLayout := Mxx.Toolkit.matrixAdd_layout
    (Mxx.matrixMultiply signalCoefficient targetNoise)
    (Mxx.matrixMultiply inputNoise subject) signalNoiseLayout inputNoiseSubjectLayout
  have rightLayout := Mxx.Toolkit.matrixAdd_layout
    (Mxx.matrixMultiply
      (Mxx.matrixMultiply signalCoefficient targetCoefficient) targetBasis)
    (Mxx.matrixAdd (Mxx.matrixMultiply signalCoefficient targetNoise)
      (Mxx.matrixMultiply inputNoise subject)) signalLayout noiseLayout
  apply Mxx.Toolkit.modEq_of_matrixValue_eq q ringDimension outputRows columns
    _ _ leftLayout rightLayout
  exact affineRelationProduct_matrixValue q ringDimension outputRows basisRows inner
    targetInner columns signalCoefficient basis inputNoise subject target targetCoefficient
    targetBasis targetNoise signalCoefficientLayout basisLayout inputNoiseLayout subjectLayout
    targetLayout targetCoefficientLayout targetBasisLayout targetNoiseLayout basisRelation
    targetRelation

/-- The opaque noise introduced by the affine relation rewrite is bounded by the sum of the two
existing worst-case product bounds.  This is deterministic hard-bound arithmetic: it uses neither
independence nor a CLT assumption. -/
theorem affineRelationProduct_noise_norm_le
    (q ringDimension outputRows basisRows inner columns
      signalBound targetNoiseBound inputNoiseBound subjectBound : Nat)
    [NeZero q]
    (signalCoefficient targetNoise inputNoise subject : Mxx.Matrix)
    (signalCoefficientLayout : Mxx.Toolkit.MatrixLayout signalCoefficient q ringDimension
      outputRows basisRows)
    (targetNoiseLayout : Mxx.Toolkit.MatrixLayout targetNoise q ringDimension basisRows columns)
    (inputNoiseLayout : Mxx.Toolkit.MatrixLayout inputNoise q ringDimension outputRows inner)
    (subjectLayout : Mxx.Toolkit.MatrixLayout subject q ringDimension inner columns)
    (signalNorm : Mxx.maxCenteredCoefficientNorm signalCoefficient ≤ signalBound)
    (targetNoiseNorm : Mxx.maxCenteredCoefficientNorm targetNoise ≤ targetNoiseBound)
    (inputNoiseNorm : Mxx.maxCenteredCoefficientNorm inputNoise ≤ inputNoiseBound)
    (subjectNorm : Mxx.maxCenteredCoefficientNorm subject ≤ subjectBound) :
    Mxx.maxCenteredCoefficientNorm
        (Mxx.matrixAdd
          (Mxx.matrixMul signalCoefficient targetNoise)
          (Mxx.matrixMul inputNoise subject)) ≤
      ringDimension * basisRows * signalBound * targetNoiseBound +
        ringDimension * inner * inputNoiseBound * subjectBound := by
  apply le_trans (Mxx.Toolkit.matrixAdd_norm_le q
    (Mxx.matrixMul signalCoefficient targetNoise)
    (Mxx.matrixMul inputNoise subject)
    (Mxx.Toolkit.matrixMul_layout signalCoefficient targetNoise
      signalCoefficientLayout targetNoiseLayout).modulus
    (Mxx.Toolkit.matrixMul_layout inputNoise subject inputNoiseLayout subjectLayout).modulus)
  exact Nat.add_le_add
    (Mxx.Toolkit.matrixMul_norm_le q ringDimension basisRows signalBound targetNoiseBound
      signalCoefficient targetNoise signalCoefficientLayout.modulus targetNoiseLayout.modulus
      signalCoefficientLayout.ringDimension targetNoiseLayout.ringDimension
      signalCoefficientLayout.columns targetNoiseLayout.rows signalNorm targetNoiseNorm)
    (Mxx.Toolkit.matrixMul_norm_le q ringDimension inner inputNoiseBound subjectBound
      inputNoise subject inputNoiseLayout.modulus subjectLayout.modulus
      inputNoiseLayout.ringDimension subjectLayout.ringDimension
      inputNoiseLayout.columns subjectLayout.rows inputNoiseNorm subjectNorm)

end Mxx.Certificate
