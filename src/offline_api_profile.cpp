#define main server_entrypoint_for_offline_profile
#include "server.cpp"
#undef main

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {

[[noreturn]] void bench_fatal(const char* msg) {
    std::cerr << msg << "\n";
    std::_Exit(1);
}

std::string read_all(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) bench_fatal("cannot open input");
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

size_t object_end(std::string_view s, size_t open) {
    int depth = 0;
    bool in_string = false;
    bool esc = false;
    for (size_t i = open; i < s.size(); ++i) {
        char c = s[i];
        if (in_string) {
            if (esc) esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"') in_string = false;
        } else {
            if (c == '"') in_string = true;
            else if (c == '{') ++depth;
            else if (c == '}' && --depth == 0) return i + 1;
        }
    }
    bench_fatal("unterminated object");
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: offline-api-profile index.bin test-data.json\n";
        return 2;
    }

    silent::MappedIndex index(argv[1]);
    SearchConfig cfg;
    cfg.nprobe = std::getenv("NPROBE") ? std::atoi(std::getenv("NPROBE")) : 2;
    cfg.fast_nprobe = std::getenv("FAST_NPROBE") ? std::atoi(std::getenv("FAST_NPROBE")) : 0;
    cfg.repair_min = std::getenv("REPAIR_MIN") ? std::atoi(std::getenv("REPAIR_MIN")) : 1;
    cfg.repair_max = std::getenv("REPAIR_MAX") ? std::atoi(std::getenv("REPAIR_MAX")) : 4;
    cfg.adaptive_min = std::getenv("ADAPTIVE_MIN") ? std::atoi(std::getenv("ADAPTIVE_MIN")) : 1;
    cfg.adaptive_max = std::getenv("ADAPTIVE_MAX") ? std::atoi(std::getenv("ADAPTIVE_MAX")) : 4;

    std::string json = read_all(argv[2]);
    std::string_view all(json);
    std::vector<double> per_request_us;
    per_request_us.reserve(60000);

    uint32_t fp = 0;
    uint32_t fn = 0;
    uint32_t parse_errors = 0;
    uint32_t total = 0;
    auto t0 = Clock::now();
    size_t p = 0;
    while ((p = all.find("\"request\"", p)) != std::string_view::npos) {
        size_t open = all.find('{', p);
        size_t end = object_end(all, open);
        std::string_view req = all.substr(open, end - open);

        size_t exp = all.find("\"expected_approved\"", end);
        size_t colon = all.find(':', exp);
        size_t val = all.find_first_not_of(" \n\r\t", colon + 1);
        bool expected = all.substr(val, 4) == "true";

        auto q0 = Clock::now();
        int16_t q[Dims];
        uint8_t fraud = 0;
        if (!vectorize_fast(req, q)) {
            ++parse_errors;
        } else {
            fraud = search_fraud(index, q, cfg);
        }
        per_request_us.push_back(std::chrono::duration<double, std::micro>(Clock::now() - q0).count());

        bool approved = fraud < 3;
        if (approved != expected) {
            if (approved) ++fn;
            else ++fp;
        }
        ++total;
        p = end;
    }

    double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    std::sort(per_request_us.begin(), per_request_us.end());
    auto pct = [&](double pctl) {
        size_t idx = std::min(per_request_us.size() - 1, static_cast<size_t>(pctl * double(per_request_us.size() - 1)));
        return per_request_us[idx];
    };

    std::cout << "offline_api total=" << total
              << " total_ms=" << ms
              << " avg_us=" << (ms * 1000.0 / double(total))
              << " p50_us=" << pct(0.50)
              << " p95_us=" << pct(0.95)
              << " p99_us=" << pct(0.99)
              << " max_us=" << per_request_us.back()
              << " fp=" << fp
              << " fn=" << fn
              << " parse_errors=" << parse_errors
              << "\n";
    return 0;
}
