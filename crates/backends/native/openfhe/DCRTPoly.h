#pragma once

#include <cstdint>
#include "openfhe/core/lattice/hal/lat-backend.h"
#include "rust/cxx.h"
#include "openfhe/core/math/matrix.h"
#include "openfhe/core/utils/serial.h"

namespace openfhe
{

    using Matrix = lbcrypto::Matrix<lbcrypto::DCRTPoly>;

    class DCRTPoly final
    {
        lbcrypto::DCRTPoly m_poly;

    public:
        DCRTPoly(lbcrypto::DCRTPoly &&poly) noexcept;
        DCRTPoly(const DCRTPoly &) = delete;
        DCRTPoly(DCRTPoly &&) = delete;
        DCRTPoly &operator=(const DCRTPoly &) = delete;
        DCRTPoly &operator=(DCRTPoly &&) = delete;

        [[nodiscard]] const lbcrypto::DCRTPoly &GetPoly() const noexcept;
        [[nodiscard]] rust::String GetString() const;
        [[nodiscard]] bool IsEqual(const DCRTPoly &other) const noexcept;
        [[nodiscard]] rust::Vec<rust::String> GetCoefficients() const;
        [[nodiscard]] rust::Vec<rust::u8> GetCoefficientsBytes() const;
        [[nodiscard]] rust::Vec<rust::u8> GetEvaluationBytes() const;
        [[nodiscard]] rust::String GetModulus() const;
        [[nodiscard]] std::unique_ptr<DCRTPoly> Clone() const;
        [[nodiscard]] std::unique_ptr<DCRTPoly> Negate() const;
        [[nodiscard]] std::unique_ptr<Matrix> Decompose(uint32_t baseBits) const;
    };

    // Generator functions
    [[nodiscard]] std::unique_ptr<DCRTPoly> DCRTPolyGenFromConst(
        usint n,
        size_t size,
        size_t kRes,
        rust::Slice<const uint64_t> value_limbs);

    [[nodiscard]] std::unique_ptr<DCRTPoly> DCRTPolyGenFromVec(
        usint n,
        size_t size,
        size_t kRes,
        rust::Slice<const uint64_t> values_limbs,
        size_t limbs_per_int);

    // Generate a polynomial from integers interpreted as EVALUATION slots,
    // and return the resulting DCRTPoly in EVALUATION format.
    [[nodiscard]] std::unique_ptr<DCRTPoly> DCRTPolyGenFromEvalVec(
        usint n,
        size_t size,
        size_t kRes,
        rust::Slice<const uint64_t> values_limbs,
        size_t limbs_per_int);

    [[nodiscard]] std::unique_ptr<DCRTPoly> DCRTPolyGenFromBug(usint n, size_t size, size_t kRes);
    [[nodiscard]] std::unique_ptr<DCRTPoly> DCRTPolyGenFromDug(usint n, size_t size, size_t kRes);
    [[nodiscard]] std::unique_ptr<DCRTPoly> DCRTPolyGenFromDgg(usint n, size_t size, size_t kRes, double sigma);
    [[nodiscard]] std::unique_ptr<DCRTPoly> DCRTPolyGenFromTug(usint n, size_t size, size_t kRes);

    // Arithmetic
    [[nodiscard]] std::unique_ptr<DCRTPoly> DCRTPolyAdd(const DCRTPoly &rhs, const DCRTPoly &lhs);
    [[nodiscard]] std::unique_ptr<DCRTPoly> DCRTPolyMul(const DCRTPoly &rhs, const DCRTPoly &lhs);

    class DCRTPolyParams final
    {
        std::shared_ptr<lbcrypto::DCRTPoly::Params> m_params;

    public:
        DCRTPolyParams() = default;
        DCRTPolyParams(const std::shared_ptr<lbcrypto::DCRTPoly::Params> &params) noexcept;
        DCRTPolyParams(const DCRTPolyParams &) = delete;
        DCRTPolyParams(DCRTPolyParams &&) = delete;
        DCRTPolyParams &operator=(const DCRTPolyParams &) = delete;
        DCRTPolyParams &operator=(DCRTPolyParams &&) = delete;

        [[nodiscard]] const std::shared_ptr<lbcrypto::DCRTPoly::Params> &GetRef() const noexcept;
    };

    // Generator functions
    [[nodiscard]] std::unique_ptr<DCRTPolyParams> DCRTPolyGenNullParams();

    // Matrix functions
    [[nodiscard]] std::unique_ptr<Matrix> MatrixGen(
        usint n,
        size_t size,
        size_t kRes,
        size_t nrow,
        size_t ncol);

    std::unique_ptr<Matrix> GetMatrixFromFs(
        usint n,
        size_t size,
        size_t kRes,
        const rust::String &path);

    void SetMatrixElement(
        Matrix &matrix,
        size_t row,
        size_t col,
        const DCRTPoly &element);

    [[nodiscard]] std::unique_ptr<DCRTPoly> GetMatrixElement(
        const Matrix &matrix,
        size_t row,
        size_t col);

    size_t GetMatrixRows(const Matrix &matrix);
    size_t GetMatrixCols(const Matrix &matrix);
    std::unique_ptr<Matrix> ExtractMatrixRow(
        const Matrix &matrix,
        size_t row);
    std::unique_ptr<Matrix> ExtractMatrixCol(
        const Matrix &matrix,
        size_t col);
    std::unique_ptr<Matrix> ExtractMatrixRows(
        const Matrix &matrix,
        size_t startRow,
        size_t endRow);
    std::unique_ptr<Matrix> ExtractMatrixCols(
        const Matrix &matrix,
        size_t startCol,
        size_t endCol);

    void FormatMatrixCoefficient(
        Matrix &matrix);
} // openfhe
