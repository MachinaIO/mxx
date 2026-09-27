// Standalone comparison driver for PhantomFHE revision pinned by build_phantom_gpu.sh.
// Upstream implementation is linked externally and is not vendored here.
#include "phantom.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

using namespace phantom;
using namespace phantom::arith;

static void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
static void array(std::ostream &out, const std::vector<uint64_t> &values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << values[i];
    out << ']';
}
static std::vector<uint64_t> read_array(const std::string &text, const std::string &key) {
    auto begin = text.find('"' + key + '"');
    if (begin == std::string::npos) throw std::runtime_error("Missing manifest array: " + key);
    begin = text.find('[', begin);
    auto end = text.find(']', begin);
    if (begin == std::string::npos || end == std::string::npos) throw std::runtime_error("Invalid manifest array");
    std::string values = text.substr(begin + 1, end - begin - 1);
    std::replace(values.begin(), values.end(), ',', ' ');
    std::istringstream input(values);
    std::vector<uint64_t> result;
    uint64_t value;
    while (input >> value) result.push_back(value);
    return result;
}
static double read_number(const std::string &text, const std::string &key) {
    auto begin = text.find('"' + key + '"');
    if (begin == std::string::npos) throw std::runtime_error("Missing manifest number: " + key);
    begin = text.find(':', begin);
    return std::stod(text.substr(begin + 1));
}
int main(int argc, char **argv) try {
    if (argc < 3 || argc > 5) throw std::runtime_error("Usage: phantom_gpu_bgv {54|36} MANIFEST [REPEATS=100] [manifest-only]");
    const std::string preset = argv[1];
    if (preset != "54" && preset != "36") throw std::runtime_error("Unknown preset");
    if(argc==5 && std::string(argv[4])!="manifest-only") throw std::runtime_error("Unknown mode");
    const bool wide = preset == "54";
    const size_t n = 8192, p_size = wide ? 1 : 2;
    const int repeats = argc > 3 ? std::stoi(argv[3]) : 100;
    if (repeats <= 0) throw std::runtime_error("REPEATS must be positive");
    auto moduli = CoeffModulus::Create(n, wide ? std::vector<int>{54,54,54,56} : std::vector<int>{36,36,36,36,37,37});
    auto plain_modulus = PlainModulus::Batching(n, 20);
    const uint64_t t = plain_modulus.value();
    std::vector<uint64_t> q, p, x(n), y(n);
    for (size_t i=0; i<moduli.size(); ++i) (i<moduli.size()-p_size ? q : p).push_back(moduli[i].value());
    std::ifstream existing(argv[2]);
    if (existing) {
        std::string text((std::istreambuf_iterator<char>(existing)), {});
        if (read_number(text,"n")!=n || read_number(text,"t")!=t || read_number(text,"digit_size")!=p_size || read_number(text,"sigma")!=3.2) throw std::runtime_error("Manifest scalar parameters differ from preset");
        if (read_array(text,"q") != q || read_array(text,"p") != p) throw std::runtime_error("Manifest basis differs from preset");
        x = read_array(text,"x"); y = read_array(text,"y");
        if (x.size()!=n || y.size()!=n) throw std::runtime_error("Invalid manifest slot count");
        for (auto value : x) if(value>=t) throw std::runtime_error("Input outside plaintext modulus");
        for (auto value : y) if(value>=t) throw std::runtime_error("Input outside plaintext modulus");
    } else {
        std::random_device random;
        std::mt19937_64 generator(random());
        std::uniform_int_distribution<uint64_t> distribution(0,t-1);
        for(size_t i=0;i<n;++i) {x[i]=distribution(generator);y[i]=distribution(generator);}
        std::ofstream out(argv[2]);
        out << "{\"n\":" << n << ",\"t\":" << t << ",\"digit_size\":" << p_size << ",\"sigma\":3.2,\"q\":";
        array(out,q); out << ",\"p\":";array(out,p);out<<",\"x\":";array(out,x);out<<",\"y\":";array(out,y);out<<"}\n";
        if(!out) throw std::runtime_error("Cannot write manifest");
    }
    if(argc==5 && std::string(argv[4])=="manifest-only") return 0;
    EncryptionParameters parms(scheme_type::bgv);
    parms.set_poly_modulus_degree(n);parms.set_coeff_modulus(moduli);
    parms.set_special_modulus_size(p_size);parms.set_plain_modulus(plain_modulus);
    PhantomContext context(parms);
    PhantomSecretKey secret(context);
    auto relin=secret.gen_relinkey(context);
    PhantomBatchEncoder encoder(context);
    PhantomPlaintext px,py;
    encoder.encode(context,x,px);encoder.encode(context,y,py);
    // Both implementations use public-key fresh encryption; no key generation is timed.
    auto public_key=secret.gen_publickey(context);
    PhantomCiphertext cx,cy;
    public_key.encrypt_asymmetric(context,px,cx);
    public_key.encrypt_asymmetric(context,py,cy);
    // Stage-only inputs are prepared once, before warmup and measurement.
    PhantomCiphertext quadratic(cx);
    multiply_inplace(context,quadratic,cy);
    PhantomCiphertext relinearized(quadratic);
    relinearize_inplace(context,relinearized,relin);
    std::vector<uint64_t> expected(n);
    for(size_t i=0;i<n;++i) expected[i]=x[i]*y[i]%t;
    cudaEvent_t start,stop;
    check(cudaEventCreate(&start));check(cudaEventCreate(&stop));
    cudaDeviceProp device{};int device_id;check(cudaGetDevice(&device_id));check(cudaGetDeviceProperties(&device,device_id));
    std::cout.precision(17);
    std::cout << "{\"library\":\"PhantomFHE\",\"revision\":\"1f4a198443b3af77118e51f53d5b8f332154b875\",\"device\":\"" << device.name
      << "\",\"error_distribution\":\"centered_binomial_21\",\"actual_sigma\":" << std::sqrt(10.5)
      << ",\"secret_distribution\":\"uniform_ternary\",\"n\":" << n << ",\"t\":" << t << ",\"digit_size\":" << p_size << ",\"q\":";
    array(std::cout,q); std::cout << ",\"p\":"; array(std::cout,p);
    std::cout << ",\"x\":"; array(std::cout,x); std::cout << ",\"y\":"; array(std::cout,y);
    std::cout << ",\"samples\":[";
    bool first=true;
    for(const std::string operation : {"multiply_relinearize", "multiply_relinearize_modswitch",
                                      "multiply", "relinearize", "modswitch"}) {
        std::vector<double> walls,gpus;
        const bool do_multiply = operation == "multiply" || operation == "multiply_relinearize" ||
                                 operation == "multiply_relinearize_modswitch";
        const bool do_relinearize = operation == "relinearize" || operation == "multiply_relinearize" ||
                                   operation == "multiply_relinearize_modswitch";
        const bool do_modswitch = operation == "modswitch" || operation == "multiply_relinearize_modswitch";
        const PhantomCiphertext &input = operation == "relinearize" ? quadratic :
                                          operation == "modswitch" ? relinearized : cx;
        for(int iteration=-1;iteration<repeats;++iteration) {
            PhantomCiphertext result(input);
            // Finish the input copy before either timer starts; no device-wide fence.
            check(cudaEventRecord(stop,cudaStreamPerThread));check(cudaEventSynchronize(stop));
            auto begin=std::chrono::steady_clock::now();
            check(cudaEventRecord(start,cudaStreamPerThread));
            if(do_multiply)
                multiply_inplace(context,result,cy);
            if(do_relinearize)
                relinearize_inplace(context,result,relin);
            if(do_modswitch)
                mod_switch_to_next_inplace(context,result);
            check(cudaEventRecord(stop,cudaStreamPerThread));check(cudaEventSynchronize(stop));
            auto end=std::chrono::steady_clock::now();
            float elapsed_ms;check(cudaEventElapsedTime(&elapsed_ms,start,stop));
            double wall=std::chrono::duration<double>(end-begin).count(), gpu=elapsed_ms/1000.0;
            // Validate every measured sample, outside the timed region.
            PhantomPlaintext decoded;secret.decrypt(context,result,decoded);
            std::vector<uint64_t> actual;encoder.decode(context,decoded,actual);
            if(actual!=expected) throw std::runtime_error("BGV round trip mismatch");
            if(iteration<0) continue;
            walls.push_back(wall);gpus.push_back(gpu);
            std::cout<<(first?"":",")<<"{\"operation\":\""<<operation
                <<"\",\"iteration\":"<<iteration<<",\"wall_seconds\":"<<wall<<",\"gpu_seconds\":"<<gpu<<"}";first=false;
        }
        auto median=[](std::vector<double> v){std::sort(v.begin(),v.end());return (v[(v.size()-1)/2]+v[v.size()/2])/2;};
        std::cerr << operation
          << " wall_median_seconds="<<median(walls)<<" gpu_median_seconds="<<median(gpus)
          <<" wall_mean_seconds="<<std::accumulate(walls.begin(),walls.end(),0.0)/walls.size()
          <<" gpu_mean_seconds="<<std::accumulate(gpus.begin(),gpus.end(),0.0)/gpus.size()<<"\n";
    }
    std::cout<<"],\"correct\":true}\n";
    check(cudaEventDestroy(start));check(cudaEventDestroy(stop));
    return 0;
} catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
