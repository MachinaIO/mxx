// Sampling adapters preserve the upstream openfhe-rs algorithms (revision
// 9c9d81c, BSD-2-Clause; see openfhe/LICENSE), but derive parameters from the
// actual input basis instead of regenerating them from depth and bit width.
#include "ExactBasis.h"
#include "openfhe/core/math/nbtheory.h"
#include "openfhe/core/lattice/trapdoor.h"
#include <map>
#include <tuple>
#include <sstream>

namespace mxx {
namespace {
// OpenFHE's process-global transform cache is indexed only by modulus. Two
// rings sharing a prime can replace each other's tables while a transform is
// reading them. Keep immutable tables per thread and complete ring identity,
// and call the existing stateless OpenFHE butterfly implementation directly.
struct TransformTables {
    lbcrypto::NativeVector forward, inverse, forward_precon, inverse_precon;
    lbcrypto::NativeInteger dimension_inverse, dimension_inverse_precon;
};

void transform_format(lbcrypto::DCRTPoly &polynomial, Format destination) {
    if (polynomial.GetFormat() == destination) return;
    const auto dimension = polynomial.GetRingDimension();
    if (dimension == 1) {
        for (auto &tower : polynomial.GetAllElements()) tower.OverrideFormat(destination);
        polynomial.OverrideFormat(destination);
        return;
    }
    using Key = std::tuple<uint32_t, uint64_t, uint64_t>;
    auto &towers = polynomial.GetAllElements();
#pragma omp parallel for num_threads(lbcrypto::OpenFHEParallelControls.GetThreadLimit(towers.size()))
    for (size_t tower_index = 0; tower_index < towers.size(); ++tower_index) {
        thread_local std::map<Key, TransformTables> tables_by_ring;
        thread_local size_t cached_coefficients = 0;
        auto &tower = towers[tower_index];
        const auto modulus = tower.GetModulus();
        const auto root = tower.GetRootOfUnity();
        const Key key(dimension, modulus.ConvertToInt(), root.ConvertToInt());
        auto found = tables_by_ring.find(key);
        if (found == tables_by_ring.end()) {
            // Retain at most 262144 native table words per worker, with at most 64
            // distinct rings. Larger transforms use a temporary table only.
            if (tables_by_ring.size() >= 64 || cached_coefficients + dimension > 65536) {
                tables_by_ring.clear();
                cached_coefficients = 0;
            }
            TransformTables tables{
                lbcrypto::NativeVector(dimension, modulus), lbcrypto::NativeVector(dimension, modulus),
                lbcrypto::NativeVector(dimension, modulus), lbcrypto::NativeVector(dimension, modulus),
                lbcrypto::NativeInteger(dimension).ModInverse(modulus), lbcrypto::NativeInteger(0)};
            tables.dimension_inverse_precon = tables.dimension_inverse.PrepModMulConst(modulus);
            const auto inverse_root = root.ModInverse(modulus);
            const auto mu = modulus.ComputeMu();
            lbcrypto::NativeInteger forward(1), inverse(1);
            const auto bits = lbcrypto::GetMSB(dimension - 1);
            for (uint32_t index = 0; index < dimension; ++index) {
                const auto reversed = lbcrypto::ReverseBits(index, bits);
                tables.forward[reversed] = forward;
                tables.inverse[reversed] = inverse;
                tables.forward_precon[reversed] = forward.PrepModMulConst(modulus);
                tables.inverse_precon[reversed] = inverse.PrepModMulConst(modulus);
                forward.ModMulEq(root, modulus, mu);
                inverse.ModMulEq(inverse_root, modulus, mu);
            }
            found = tables_by_ring.emplace(key, std::move(tables)).first;
            cached_coefficients += dimension;
        }
        const auto &tables = found->second;
        auto values = tower.GetValues();
        intnat::NumberTheoreticTransformNat<lbcrypto::NativeVector> transform;
        if (destination == Format::EVALUATION) {
            transform.ForwardTransformToBitReverseInPlace(tables.forward, tables.forward_precon, &values);
        } else {
            transform.InverseTransformFromBitReverseInPlace(tables.inverse, tables.inverse_precon,
                tables.dimension_inverse, tables.dimension_inverse_precon, &values);
        }
        tower.SetValues(std::move(values), destination);
        if (dimension > 65536) {
            tables_by_ring.clear();
            cached_coefficients = 0;
        }
    }
    polynomial.OverrideFormat(destination);
}
}

std::unique_ptr<openfhe::DCRTPoly> exact_basis_matrix_entry(
    const openfhe::Matrix &matrix, size_t row, size_t column) {
    if (row >= matrix.GetRows() || column >= matrix.GetCols())
        throw std::invalid_argument("matrix entry out of bounds");
    // Upstream GetMatrixElement performs CRT interpolation and switches format
    // through process-global NTT caches. Matrix entries already have exact
    // residues: preserve their basis, roots, and format by copying directly.
    return std::make_unique<openfhe::DCRTPoly>(lbcrypto::DCRTPoly(matrix(row, column)));
}

void exact_basis_matrix_coefficients(openfhe::Matrix &matrix) {
#pragma omp parallel for collapse(2)
    for (size_t row = 0; row < matrix.GetRows(); ++row)
        for (size_t column = 0; column < matrix.GetCols(); ++column)
            transform_format(matrix(row, column), Format::COEFFICIENT);
}

std::unique_ptr<openfhe::DCRTPoly> exact_basis_bgv_mod_reduce(
    const openfhe::DCRTPoly &input, uint64_t plaintext_modulus) {
    auto polynomial = input.GetPoly();
    if (polynomial.GetNumOfElements() < 2 || plaintext_modulus < 2)
        throw std::invalid_argument("BGV ModReduce requires two CRT limbs and plaintext modulus >= 2");
    const auto format = polynomial.GetFormat();
    // Use the thread-local transforms above instead of OpenFHE's global cache.
    transform_format(polynomial, Format::COEFFICIENT);
    const lbcrypto::NativeInteger t(plaintext_modulus);
    const auto p = polynomial.GetAllElements().back().GetModulus();
    const auto negative_inverse = p - t.Mod(p).ModInverse(p);
    std::vector<lbcrypto::NativeInteger> t_precon, p_inverse, p_inverse_precon;
    for (size_t index = 0; index + 1 < polynomial.GetNumOfElements(); ++index) {
        const auto q = polynomial.GetElementAtIndex(index).GetModulus();
        t_precon.push_back(t.Mod(q).PrepModMulConst(q));
        p_inverse.push_back(p.Mod(q).ModInverse(q));
        p_inverse_precon.push_back(p_inverse.back().PrepModMulConst(q));
    }
    polynomial.ModReduce(t, t_precon, negative_inverse,
        negative_inverse.PrepModMulConst(p), p_inverse, p_inverse_precon);
    transform_format(polynomial, format);
    return std::make_unique<openfhe::DCRTPoly>(std::move(polynomial));
}

rust::Vec<uint8_t> exact_basis_coefficients(const openfhe::DCRTPoly &input) {
    auto polynomial = input.GetPoly();
    transform_format(polynomial, Format::COEFFICIENT);
    const auto interpolated = polynomial.CRTInterpolate();
    std::stringstream stream;
    lbcrypto::Serial::Serialize(interpolated.GetValues(), stream, lbcrypto::SerType::BINARY);
    const auto bytes = stream.str();
    rust::Vec<uint8_t> output;
    output.reserve(bytes.size());
    for (const auto byte : bytes) output.push_back(static_cast<uint8_t>(byte));
    return output;
}

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

#pragma omp parallel for collapse(2)
        for (size_t row = 0; row < p1.GetRows(); ++row)
            for (size_t column = 0; column < p1.GetCols(); ++column)
                transform_format(p1(row, column), Format::EVALUATION);

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

namespace {
struct RnsPlan {
    std::shared_ptr<lbcrypto::DCRTPoly::Params> parameters;
    std::vector<std::vector<size_t>> groups;
    std::vector<uint64_t> scales, weights, inverses;
    std::vector<size_t> retained;
};
uint64_t rns_mul(uint64_t a, uint64_t b, uint64_t modulus) {
    return static_cast<uint64_t>((static_cast<unsigned __int128>(a) * b) % modulus);
}
uint64_t rns_inverse(uint64_t a, uint64_t modulus) {
    return lbcrypto::NativeInteger(a).ModInverse(lbcrypto::NativeInteger(modulus)).ConvertToInt();
}
}

std::unique_ptr<openfhe::Matrix> exact_basis_rns(
    const openfhe::DCRTPoly &input, rust::Slice<const uint64_t> moduli,
    size_t digit_size, bool normalize, uint64_t plaintext_modulus) {
    auto source = input.GetPoly();
    const auto dimension = source.GetRingDimension();
    const bool down = plaintext_modulus != 0;
    std::vector<uint64_t> primes;
    for (const auto &tower : source.GetAllElements()) primes.push_back(tower.GetModulus().ConvertToInt());
    if (moduli.empty() || (!down && digit_size == 0) || (down && plaintext_modulus < 2))
        throw std::invalid_argument("invalid RNS conversion parameters");
    using Key = std::tuple<uint32_t, std::vector<uint64_t>, std::vector<uint64_t>, size_t, bool, uint64_t>;
    const Key key(dimension, primes, std::vector<uint64_t>(moduli.begin(), moduli.end()), digit_size, normalize, plaintext_modulus);
    thread_local std::map<Key, RnsPlan> plans;
    auto found = plans.find(key);
    if (found == plans.end()) {
        RnsPlan plan;
        plan.parameters = parameters_for_basis(dimension, moduli);
        plan.scales.resize(primes.size());
        plan.weights.resize(primes.size() * moduli.size());
        if (down) {
            if (moduli.size() >= primes.size()) throw std::invalid_argument("RNS ModDown needs a strict subset");
            plan.groups.resize(1);
            for (size_t i = 0; i < primes.size(); ++i)
                if (std::find(moduli.begin(), moduli.end(), primes[i]) == moduli.end()) plan.groups[0].push_back(i);
            for (const auto q : moduli) {
                auto position = std::find(primes.begin(), primes.end(), q);
                if (position == primes.end()) throw std::invalid_argument("RNS ModDown destination is not a subset");
                plan.retained.push_back(position - primes.begin());
                uint64_t product = 1;
                for (const auto i : plan.groups[0]) product = rns_mul(product, primes[i] % q, q);
                plan.inverses.push_back(rns_inverse(product, q));
            }
        } else {
            for (const auto q : primes)
                if (std::find(moduli.begin(), moduli.end(), q) == moduli.end()) throw std::invalid_argument("RNS ModUp destination must contain source");
            plan.groups.resize(1 + (primes.size() - 1) / digit_size);
            for (size_t i = 0; i < primes.size(); ++i) plan.groups[i / digit_size].push_back(i);
        }
        for (const auto &group : plan.groups) {
            for (const auto i : group) {
                const uint64_t p = primes[i];
                uint64_t product = 1;
                for (const auto j : group) if (i != j) product = rns_mul(product, primes[j] % p, p);
                if (!down && normalize)
                    for (size_t j = 0; j < primes.size(); ++j)
                        if (std::find(group.begin(), group.end(), j) == group.end()) product = rns_mul(product, primes[j] % p, p);
                plan.scales[i] = rns_inverse(product, p);
                if (down) plan.scales[i] = rns_mul(plan.scales[i], p - rns_inverse(plaintext_modulus % p, p), p);
                for (size_t target = 0; target < moduli.size(); ++target) {
                    const uint64_t q = moduli[target];
                    uint64_t weight = 1;
                    for (const auto j : group) if (i != j) weight = rns_mul(weight, primes[j] % q, q);
                    plan.weights[i * moduli.size() + target] = weight;
                }
            }
        }
        // Bound cache retention per Rayon worker without shared transform/cache locks.
        if (plans.size() >= 32) plans.clear();
        found = plans.emplace(key, std::move(plan)).first;
    }
    const auto &plan = found->second;
    transform_format(source, Format::COEFFICIENT);
    auto allocator = lbcrypto::DCRTPoly::Allocator(plan.parameters, Format::COEFFICIENT);
    openfhe::Matrix result(allocator, plan.groups.size(), 1);
    // Convert once per source tower. No full-CRT coefficient reconstruction is used.
    std::vector<std::vector<uint64_t>> scaled(primes.size());
#pragma omp parallel for num_threads(lbcrypto::OpenFHEParallelControls.GetThreadLimit(primes.size()))
    for (size_t i = 0; i < primes.size(); ++i) {
        if (!plan.scales[i]) continue;
        scaled[i].resize(dimension);
        const auto &values = source.GetElementAtIndex(i).GetValues();
        for (size_t k = 0; k < dimension; ++k) scaled[i][k] = rns_mul(values[k].ConvertToInt(), plan.scales[i], primes[i]);
    }
#pragma omp parallel for collapse(2) num_threads(lbcrypto::OpenFHEParallelControls.GetThreadLimit(plan.groups.size() * moduli.size()))
    for (size_t group = 0; group < plan.groups.size(); ++group) {
        for (size_t target = 0; target < moduli.size(); ++target) {
            const uint64_t q = moduli[target];
            lbcrypto::NativeVector values(dimension, lbcrypto::NativeInteger(q));
            for (size_t k = 0; k < dimension; ++k) {
                uint64_t sum = 0;
                for (const auto i : plan.groups[group]) {
                    const uint64_t u = scaled[i][k], p = primes[i];
                    const uint64_t magnitude = (u <= p / 2 ? u : p - u) % q;
                    const uint64_t centered = u <= p / 2 || magnitude == 0 ? magnitude : q - magnitude;
                    const auto contribution = rns_mul(centered, plan.weights[i * moduli.size() + target], q);
                    sum = static_cast<uint64_t>((static_cast<unsigned __int128>(sum) + contribution) % q);
                }
                if (down) {
                    const auto retained = source.GetElementAtIndex(plan.retained[target]).GetValues()[k].ConvertToInt();
                    sum = rns_mul(sum, plaintext_modulus % q, q);
                    sum = static_cast<uint64_t>((static_cast<unsigned __int128>(sum) + retained) % q);
                    sum = rns_mul(sum, plan.inverses[target], q);
                }
                values[k] = sum;
            }
            result(group, 0).GetAllElements()[target].SetValues(std::move(values), Format::COEFFICIENT);
        }
    }

#pragma omp parallel for num_threads(lbcrypto::OpenFHEParallelControls.GetThreadLimit(plan.groups.size()))
    for (size_t group = 0; group < plan.groups.size(); ++group) transform_format(result(group, 0), Format::EVALUATION);
    return std::make_unique<openfhe::Matrix>(std::move(result));
}

std::unique_ptr<openfhe::DCRTPoly> exact_basis_convert(
    const openfhe::DCRTPoly &input, rust::Slice<const uint64_t> moduli, bool centered) {
    auto source = input.GetPoly();
    const auto dimension = source.GetRingDimension();
    std::shared_ptr<lbcrypto::DCRTPoly::Params> parameters;
    std::vector<size_t> selected;
    if (centered) {
        parameters = parameters_for_basis(dimension, moduli);
    } else {
        std::vector<lbcrypto::NativeInteger> primes, roots;
        for (const auto modulus : moduli) {
            size_t index = 0;
            const auto &towers = source.GetAllElements();
            while (index < towers.size() && towers[index].GetModulus().ConvertToInt() != modulus)
                ++index;
            if (index == towers.size()) throw std::invalid_argument("destination is not a source CRT subset");
            selected.push_back(index);
            primes.push_back(towers[index].GetModulus());
            roots.push_back(towers[index].GetRootOfUnity());
        }
        parameters = std::make_shared<lbcrypto::ILDCRTParams<lbcrypto::BigInteger>>(
            2 * dimension, primes, roots);
    }
    if (centered && source.GetNumOfElements() != 1)
        throw std::invalid_argument("centered rebase requires one source CRT limb");
    if (centered) transform_format(source, Format::COEFFICIENT);
    lbcrypto::DCRTPoly output(parameters, source.GetFormat(), true);
#pragma omp parallel for num_threads(lbcrypto::OpenFHEParallelControls.GetThreadLimit(moduli.size()))
    for (size_t index = 0; index < moduli.size(); ++index) {
        if (centered) {
            const auto &tower = source.GetElementAtIndex(0);
            const uint64_t p = tower.GetModulus().ConvertToInt(), q = moduli[index];
            lbcrypto::NativeVector values(dimension, lbcrypto::NativeInteger(q));
            for (size_t coefficient = 0; coefficient < dimension; ++coefficient) {
                const uint64_t u = tower.GetValues()[coefficient].ConvertToInt();
                const uint64_t magnitude = (u <= p / 2 ? u : p - u) % q;
                values[coefficient] = u <= p / 2 || magnitude == 0 ? magnitude : q - magnitude;
            }
            output.GetAllElements()[index].SetValues(std::move(values), Format::COEFFICIENT);
        } else {
            // Copy the native tower verbatim, preserving its root and NTT format.
            output.SetElementAtIndex(index, source.GetElementAtIndex(selected[index]));
        }
    }
    if (centered) transform_format(output, Format::EVALUATION);
    return std::make_unique<openfhe::DCRTPoly>(std::move(output));
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
    transform_format(polynomial, Format::EVALUATION);
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
            polynomial = lbcrypto::DCRTPoly(generator, parameters, Format::COEFFICIENT);
            break;
        }
        case 2: {
            lbcrypto::DCRTPoly::BugType generator;
            polynomial = lbcrypto::DCRTPoly(generator, parameters, Format::COEFFICIENT);
            break;
        }
        case 3: {
            lbcrypto::DCRTPoly::TugType generator;
            polynomial = lbcrypto::DCRTPoly(generator, parameters, Format::COEFFICIENT);
            break;
        }
        default: throw std::invalid_argument("invalid exact CRT sampling distribution");
    }
    transform_format(polynomial, Format::EVALUATION);
    return std::make_unique<openfhe::DCRTPoly>(std::move(polynomial));
}

rust::Vec<int64_t> exact_basis_gauss_gq(
    const openfhe::DCRTPoly &syndrome, double c, size_t digits_count,
    int64_t base, double sigma, size_t tower) {
    auto polynomial = syndrome.GetPoly();
    transform_format(polynomial, Format::COEFFICIENT);
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
