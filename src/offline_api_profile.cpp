#define main server_entrypoint_for_offline_profile
#include "server.cpp"
#undef main

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
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

struct LatencyBucket {
    std::vector<double> total_us;
    std::vector<double> search_us;
    uint64_t initial_scanned = 0;
    uint64_t initial_pruned = 0;
    uint64_t repair_scanned = 0;
    uint64_t repair_pruned = 0;

    void reserve(size_t n) {
        total_us.reserve(n);
        search_us.reserve(n);
    }

    void add(double total, double search, const silent::SearchStats& stats) {
        total_us.push_back(total);
        search_us.push_back(search);
        initial_scanned += stats.initial_scanned;
        initial_pruned += stats.initial_pruned;
        repair_scanned += stats.repair_scanned;
        repair_pruned += stats.repair_pruned;
    }
};

double percentile_sorted(const std::vector<double>& values, double pctl) {
    if (values.empty()) return 0.0;
    size_t idx = std::min(values.size() - 1, static_cast<size_t>(pctl * double(values.size() - 1)));
    return values[idx];
}

void sort_bucket(LatencyBucket& bucket) {
    std::sort(bucket.total_us.begin(), bucket.total_us.end());
    std::sort(bucket.search_us.begin(), bucket.search_us.end());
}

void print_bucket(const char* name, const LatencyBucket& bucket) {
    const double total_sum = std::accumulate(bucket.total_us.begin(), bucket.total_us.end(), 0.0);
    const double search_sum = std::accumulate(bucket.search_us.begin(), bucket.search_us.end(), 0.0);
    const double count = double(bucket.total_us.size());
    std::cout << " " << name
              << "_count=" << bucket.total_us.size()
              << " " << name << "_avg_total_us=" << (count > 0 ? total_sum / count : 0.0)
              << " " << name << "_p95_total_us=" << percentile_sorted(bucket.total_us, 0.95)
              << " " << name << "_p99_total_us=" << percentile_sorted(bucket.total_us, 0.99)
              << " " << name << "_max_total_us=" << (bucket.total_us.empty() ? 0.0 : bucket.total_us.back())
              << " " << name << "_avg_search_us=" << (count > 0 ? search_sum / count : 0.0)
              << " " << name << "_p95_search_us=" << percentile_sorted(bucket.search_us, 0.95)
              << " " << name << "_p99_search_us=" << percentile_sorted(bucket.search_us, 0.99)
              << " " << name << "_max_search_us=" << (bucket.search_us.empty() ? 0.0 : bucket.search_us.back())
              << " " << name << "_avg_initial_scanned=" << (count > 0 ? double(bucket.initial_scanned) / count : 0.0)
              << " " << name << "_avg_initial_pruned=" << (count > 0 ? double(bucket.initial_pruned) / count : 0.0)
              << " " << name << "_avg_repair_scanned=" << (count > 0 ? double(bucket.repair_scanned) / count : 0.0)
              << " " << name << "_avg_repair_pruned=" << (count > 0 ? double(bucket.repair_pruned) / count : 0.0);
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
    bool breakdown = std::getenv("PROFILE_BREAKDOWN") && std::atoi(std::getenv("PROFILE_BREAKDOWN")) != 0;

    std::string json = read_all(argv[2]);
    std::string_view all(json);
    std::vector<double> per_request_us;
    per_request_us.reserve(60000);
    LatencyBucket repair_bucket;
    LatencyBucket non_repair_bucket;
    repair_bucket.reserve(60000);
    non_repair_bucket.reserve(60000);

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
        } else if (!breakdown) {
            fraud = search_fraud(index, q, cfg);
        } else {
            auto search0 = Clock::now();
            silent::SearchStats stats;
            if (cfg.fast_nprobe > 0 && cfg.fast_nprobe < cfg.nprobe) {
                silent::SearchStats fast_stats;
                fraud = index.search(q, cfg.fast_nprobe, 99, 0, &fast_stats);
                if (fraud >= cfg.adaptive_min && fraud <= cfg.adaptive_max) {
                    fraud = index.search(q, cfg.nprobe, cfg.repair_min, cfg.repair_max, &stats);
                } else {
                    stats = fast_stats;
                }
            } else {
                fraud = index.search(q, cfg.nprobe, cfg.repair_min, cfg.repair_max, &stats);
            }
            double search_us = std::chrono::duration<double, std::micro>(Clock::now() - search0).count();
            double total_us = std::chrono::duration<double, std::micro>(Clock::now() - q0).count();
            if (stats.repair_triggered) {
                repair_bucket.add(total_us, search_us, stats);
            } else {
                non_repair_bucket.add(total_us, search_us, stats);
            }
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
    sort_bucket(repair_bucket);
    sort_bucket(non_repair_bucket);
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
              << " parse_errors=" << parse_errors;
    if (breakdown) {
        print_bucket("repair", repair_bucket);
        print_bucket("non_repair", non_repair_bucket);
    }
    std::cout << "\n";
    return 0;
}
