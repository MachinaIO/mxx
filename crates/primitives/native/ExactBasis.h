#pragma once
#include "openfhe/DCRTPoly.h"

namespace mxx {
std::unique_ptr<openfhe::DCRTPoly> exact_basis_bgv_mod_reduce(
    const openfhe::DCRTPoly &input, uint64_t plaintext_modulus);
std::unique_ptr<openfhe::DCRTPoly> exact_basis_convert(
    const openfhe::DCRTPoly &input, rust::Slice<const uint64_t> moduli, bool centered);
rust::Vec<uint8_t> exact_basis_coefficients(const openfhe::DCRTPoly &input);
void exact_basis_matrix_coefficients(openfhe::Matrix &matrix);
void exact_basis_validate(uint32_t dimension, rust::Slice<const uint64_t> moduli);
std::unique_ptr<openfhe::Matrix> exact_basis_p1(
    const openfhe::Matrix &a, const openfhe::Matrix &b, const openfhe::Matrix &d,
    const openfhe::Matrix &tp2, size_t columns, double sigma, double s, double dgg_sigma);
std::unique_ptr<openfhe::DCRTPoly> exact_basis_poly(
    uint32_t dimension, rust::Slice<const uint64_t> moduli,
    rust::Slice<const uint64_t> values, size_t limbs_per_integer, bool evaluation);
std::unique_ptr<openfhe::Matrix> exact_basis_matrix(
    uint32_t dimension, rust::Slice<const uint64_t> moduli,
    size_t rows, size_t columns, uint64_t gadget_base);
std::unique_ptr<openfhe::DCRTPoly> exact_basis_sample(
    uint32_t dimension, rust::Slice<const uint64_t> moduli, uint32_t distribution, double sigma);
rust::Vec<int64_t> exact_basis_gauss_gq(
    const openfhe::DCRTPoly &syndrome, double c, size_t digits_count,
    int64_t base, double sigma, size_t tower);
}
