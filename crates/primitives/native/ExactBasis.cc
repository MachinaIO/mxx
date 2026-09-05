// Sampling adapters preserve the upstream openfhe-rs algorithms (revision
// 9c9d81c, BSD-2-Clause; see openfhe/LICENSE), but derive parameters from the
// actual input basis instead of regenerating them from depth and bit width.
#include "ExactBasis.h"
#include "openfhe/core/math/nbtheory.h"
#include "openfhe/core/lattice/trapdoor.h"

namespace mxx {
void exact_basis_validate(uint32_t dimension, rust::Slice<const uint64_t> moduli) {
    if (moduli.empty()) throw std::invalid_argument("empty exact CRT basis");
    for (size_t index = 0; index < moduli.size(); ++index) {
        const auto modulus = moduli[index];
        if (modulus < 3 || (modulus - 1) % (2 * static_cast<uint64_t>(dimension)) ||
            !lbcrypto::MillerRabinPrimalityTest(lbcrypto::NativeInteger(modulus)))
            throw std::invalid_argument("invalid exact CRT prime");
        for (size_t earlier = 0; earlier < index; ++earlier)
            if (moduli[earlier] == modulus) throw std::invalid_argument("duplicate exact CRT prime");
    }
}
    std::unique_ptr<openfhe::Matrix> exact_basis_p1(
        const openfhe::Matrix &A,
        const openfhe::Matrix &B,
        const openfhe::Matrix &D,
        const openfhe::Matrix &tp2,
        size_t ncol,
        double sigma,
        double s,
        double dggStddev)
    {
        size_t d = A.GetRows();
        if (d == 0 || A.GetCols() == 0) throw std::invalid_argument("empty P1 covariance");
        const auto params = A(0, 0).GetParams();
        const auto n = params->GetRingDimension();


        lbcrypto::DCRTPoly::DggType dgg(dggStddev);

        auto zero_alloc = lbcrypto::DCRTPoly::Allocator(params, Format::EVALUATION);

        lbcrypto::Matrix<lbcrypto::Field2n> AF([&]()
                                               { return lbcrypto::Field2n(n, Format::EVALUATION, true); }, d, d);
        lbcrypto::Matrix<lbcrypto::Field2n> BF([&]()
                                               { return lbcrypto::Field2n(n, Format::EVALUATION, true); }, d, d);
        lbcrypto::Matrix<lbcrypto::Field2n> DF([&]()
                                               { return lbcrypto::Field2n(n, Format::EVALUATION, true); }, d, d);

        double scalarFactor = -sigma * sigma;

        for (size_t i = 0; i < d; i++)
        {
            for (size_t j = 0; j < d; j++)
            {
                AF(i, j) = lbcrypto::Field2n(A(i, j));
                AF(i, j) = AF(i, j).ScalarMult(scalarFactor);
                BF(i, j) = lbcrypto::Field2n(B(i, j));
                BF(i, j) = BF(i, j).ScalarMult(scalarFactor);
                DF(i, j) = lbcrypto::Field2n(D(i, j));
                DF(i, j) = DF(i, j).ScalarMult(scalarFactor);
                if (i == j)
                {
                    AF(i, j) = AF(i, j) + s * s;
                    DF(i, j) = DF(i, j) + s * s;
                }
            }
        }

        // converts the field elements to DFT representation
        AF.SetFormat(Format::EVALUATION);
        BF.SetFormat(Format::EVALUATION);
        DF.SetFormat(Format::EVALUATION);

        lbcrypto::Matrix<lbcrypto::Field2n> c([&]()
                                              { return lbcrypto::Field2n(n, Format::COEFFICIENT); }, 2 * d, ncol);
        double cScale = -sigma * sigma / (s * s - sigma * sigma);

#pragma omp parallel for if (ncol > 1)
        for (long jL = 0; jL < static_cast<long>(ncol); ++jL)
        {
            size_t j = static_cast<size_t>(jL);
            for (size_t i = 0; i < d; i++)
            {
                c(i, j) = lbcrypto::Field2n(tp2(i, j)).ScalarMult(cScale);
                c(i + d, j) = lbcrypto::Field2n(tp2(i + d, j)).ScalarMult(cScale);
            }
        }

        auto p1ZVector = std::make_shared<lbcrypto::Matrix<int64_t>>([]()
                                                                     { return 0; }, n * 2 * d, ncol);
        lbcrypto::LatticeGaussSampUtility<lbcrypto::DCRTPoly>::SampleMat(AF, BF, DF, c, dgg, p1ZVector);

        lbcrypto::Matrix<lbcrypto::DCRTPoly> p1(zero_alloc, 1, 1);
        std::vector<lbcrypto::Matrix<lbcrypto::DCRTPoly>> p1Cols(ncol);
#pragma omp parallel for if (ncol > 1)
        for (long jL = 0; jL < static_cast<long>(ncol); ++jL)
        {
            size_t j = static_cast<size_t>(jL);
            p1Cols[j] = lbcrypto::SplitInt64IntoElements<lbcrypto::DCRTPoly>(p1ZVector->ExtractCol(j), n, params);
        }
        if (ncol > 0)
        {
            p1 = p1Cols[0];
            for (size_t j = 1; j < ncol; ++j)
                p1.HStack(p1Cols[j]);
        }

        p1.SetFormat(Format::EVALUATION);

        return std::make_unique<openfhe::Matrix>(std::move(p1));
    }

namespace {
std::shared_ptr<lbcrypto::DCRTPoly::Params> parameters_for_basis(
    uint32_t dimension, rust::Slice<const uint64_t> moduli) {
    if (dimension < 2 || (dimension & (dimension - 1)) || moduli.empty())
        throw std::invalid_argument("invalid exact CRT ring");
    std::vector<lbcrypto::NativeInteger> primes, roots;
    for (const auto modulus : moduli) {
        primes.emplace_back(modulus);
        roots.push_back(lbcrypto::RootOfUnity<lbcrypto::NativeInteger>(2 * dimension, primes.back()));
    }
    return std::make_shared<lbcrypto::ILDCRTParams<lbcrypto::BigInteger>>(
        2 * dimension, primes, roots);
}
}

std::unique_ptr<openfhe::DCRTPoly> exact_basis_poly(
    uint32_t dimension, rust::Slice<const uint64_t> moduli,
    rust::Slice<const uint64_t> values, size_t limbs_per_integer, bool evaluation) {
    if (dimension < 2 || (dimension & (dimension - 1)) || moduli.empty())
        throw std::invalid_argument("invalid exact CRT ring");
    if (limbs_per_integer && (values.size() % limbs_per_integer ||
        values.size() / limbs_per_integer > dimension))
        throw std::invalid_argument("invalid exact CRT coefficient buffer");
    auto parameters = parameters_for_basis(dimension, moduli);
    lbcrypto::BigVector vector(dimension, parameters->GetModulus());
    if (limbs_per_integer) {
        for (size_t coefficient = 0; coefficient < values.size() / limbs_per_integer; ++coefficient) {
            lbcrypto::BigInteger integer(0);
            for (size_t limb = limbs_per_integer; limb-- > 0;)
                integer = (integer << 64) + lbcrypto::BigInteger(values[coefficient * limbs_per_integer + limb]);
            vector[coefficient] = integer.Mod(parameters->GetModulus());
        }
    }
    const auto format = evaluation ? Format::EVALUATION : Format::COEFFICIENT;
    lbcrypto::PolyImpl<lbcrypto::BigVector> large(parameters, format);
    large.SetValues(vector, format);
    lbcrypto::DCRTPoly polynomial(large, parameters);
    polynomial.SetFormat(Format::EVALUATION);
    return std::make_unique<openfhe::DCRTPoly>(std::move(polynomial));
}

std::unique_ptr<openfhe::Matrix> exact_basis_matrix(
    uint32_t dimension, rust::Slice<const uint64_t> moduli,
    size_t rows, size_t columns, uint64_t gadget_base) {
    auto parameters = parameters_for_basis(dimension, moduli);
    auto allocator = lbcrypto::DCRTPoly::Allocator(parameters, Format::EVALUATION);
    openfhe::Matrix matrix(allocator, rows, columns);
    if (gadget_base) {
        if (gadget_base < 2 || !rows || columns % (rows * moduli.size()))
            throw std::invalid_argument("invalid exact CRT gadget layout");
        const size_t digits = columns / (rows * moduli.size());
        // The repository uses one maximum-width digit stride for every tower,
        // including mixed-width and reordered bases. OpenFHE GadgetVector
        // derives this stride from only its first prime and cannot be used here.
#pragma omp parallel for if (rows > 1)
        for (long row = 0; row < static_cast<long>(rows); ++row) {
            for (size_t tower = 0; tower < moduli.size(); ++tower) {
                uint64_t power = 1;
                for (size_t digit = 0; digit < digits; ++digit) {
                    lbcrypto::NativePoly polynomial(parameters->GetParams()[tower], Format::EVALUATION, true);
                    polynomial = power;
                    matrix(row, row * (columns / rows) + tower * digits + digit)
                        .SetElementAtIndex(tower, std::move(polynomial));
                    power = static_cast<uint64_t>((static_cast<unsigned __int128>(power) * gadget_base) % moduli[tower]);
                }
            }
        }
    }
    return std::make_unique<openfhe::Matrix>(std::move(matrix));
}

std::unique_ptr<openfhe::DCRTPoly> exact_basis_sample(
    uint32_t dimension, rust::Slice<const uint64_t> moduli, uint32_t distribution, double sigma) {
    auto parameters = parameters_for_basis(dimension, moduli);
    lbcrypto::DCRTPoly polynomial;
    switch (distribution) {
        case 0: {
            lbcrypto::DCRTPoly::DugType generator;
            polynomial = lbcrypto::DCRTPoly(generator, parameters, Format::EVALUATION);
            break;
        }
        case 1: {
            lbcrypto::DCRTPoly::DggType generator(sigma);
            polynomial = lbcrypto::DCRTPoly(generator, parameters, Format::EVALUATION);
            break;
        }
        case 2: {
            lbcrypto::DCRTPoly::BugType generator;
            polynomial = lbcrypto::DCRTPoly(generator, parameters, Format::EVALUATION);
            break;
        }
        case 3: {
            lbcrypto::DCRTPoly::TugType generator;
            polynomial = lbcrypto::DCRTPoly(generator, parameters, Format::EVALUATION);
            break;
        }
        default: throw std::invalid_argument("invalid exact CRT sampling distribution");
    }
    return std::make_unique<openfhe::DCRTPoly>(std::move(polynomial));
}

rust::Vec<int64_t> exact_basis_gauss_gq(
    const openfhe::DCRTPoly &syndrome, double c, size_t digits_count,
    int64_t base, double sigma, size_t tower) {
    auto polynomial = syndrome.GetPoly();
    polynomial.SetFormat(Format::COEFFICIENT);
    const auto &component = polynomial.GetElementAtIndex(tower);
    const size_t dimension = polynomial.GetRingDimension();
    lbcrypto::DCRTPoly::DggType generator(sigma);
    lbcrypto::Matrix<int64_t> digits([]() { return 0; }, digits_count, dimension);
    lbcrypto::LatticeGaussSampUtility<lbcrypto::NativePoly>::GaussSampGqArbBase(
        component, c, digits_count, component.GetModulus(), base, generator, &digits);
    rust::Vec<int64_t> result;
    result.reserve(digits_count * dimension);
    for (size_t digit = 0; digit < digits_count; ++digit)
        for (size_t coefficient = 0; coefficient < dimension; ++coefficient)
            result.push_back(digits(digit, coefficient));
    return result;
}
}
