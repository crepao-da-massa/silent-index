#include "index.hpp"

#include <array>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {

[[noreturn]] void profile_fatal(const char* msg) {
    std::cerr << msg << "\n";
    std::_Exit(1);
}

struct Tx {
    double amount = 0;
    int installments = 0;
    std::string_view requested_at;
    double customer_avg = 1;
    int tx_count_24h = 0;
    std::string_view known_merchants;
    std::string_view merchant_id;
    std::string_view mcc;
    double merchant_avg = 0;
    bool is_online = false;
    bool card_present = false;
    double km_from_home = 0;
    bool has_last = false;
    std::string_view last_ts;
    double last_km = 0;
};

size_t find_key(std::string_view s, std::string_view key, size_t from = 0) {
    size_t p = s.find(key, from);
    if (p == std::string_view::npos) profile_fatal("missing json key");
    p = s.find(':', p + key.size());
    if (p == std::string_view::npos) profile_fatal("missing json colon");
    return p + 1;
}

double parse_number(std::string_view s, size_t& p) {
    bool neg = false;
    if (s[p] == '-') {
        neg = true;
        ++p;
    }
    double v = 0;
    while (p < s.size() && s[p] >= '0' && s[p] <= '9') {
        v = v * 10.0 + double(s[p] - '0');
        ++p;
    }
    if (p < s.size() && s[p] == '.') {
        ++p;
        double scale = 0.1;
        while (p < s.size() && s[p] >= '0' && s[p] <= '9') {
            v += double(s[p] - '0') * scale;
            scale *= 0.1;
            ++p;
        }
    }
    return neg ? -v : v;
}

double number_at(std::string_view s, std::string_view key, size_t from = 0) {
    size_t p = find_key(s, key, from);
    while (p < s.size() && (s[p] == ' ' || s[p] == '\n')) ++p;
    return parse_number(s, p);
}

int int_at(std::string_view s, std::string_view key, size_t from = 0) {
    return static_cast<int>(number_at(s, key, from));
}

std::string_view string_at(std::string_view s, std::string_view key, size_t from = 0) {
    size_t p = find_key(s, key, from);
    p = s.find('"', p);
    if (p == std::string_view::npos) profile_fatal("missing string quote");
    size_t e = s.find('"', p + 1);
    if (e == std::string_view::npos) profile_fatal("unterminated string");
    return s.substr(p + 1, e - p - 1);
}

bool bool_at(std::string_view s, std::string_view key, size_t from = 0) {
    size_t p = find_key(s, key, from);
    while (p < s.size() && s[p] == ' ') ++p;
    return s.substr(p, 4) == "true";
}

Tx parse_tx(std::string_view body) {
    Tx tx;
    size_t txp = body.find("\"transaction\"");
    size_t custp = body.find("\"customer\"");
    size_t merchp = body.find("\"merchant\"");
    size_t termp = body.find("\"terminal\"");
    size_t lastp = body.find("\"last_transaction\"");

    tx.amount = number_at(body, "\"amount\"", txp);
    tx.installments = int_at(body, "\"installments\"", txp);
    tx.requested_at = string_at(body, "\"requested_at\"", txp);
    tx.customer_avg = number_at(body, "\"avg_amount\"", custp);
    tx.tx_count_24h = int_at(body, "\"tx_count_24h\"", custp);

    size_t known_key = body.find("\"known_merchants\"", custp);
    size_t arr_start = body.find('[', known_key);
    size_t arr_end = body.find(']', arr_start);
    tx.known_merchants = body.substr(arr_start, arr_end - arr_start + 1);

    tx.merchant_id = string_at(body, "\"id\"", merchp);
    tx.mcc = string_at(body, "\"mcc\"", merchp);
    tx.merchant_avg = number_at(body, "\"avg_amount\"", merchp);
    tx.is_online = bool_at(body, "\"is_online\"", termp);
    tx.card_present = bool_at(body, "\"card_present\"", termp);
    tx.km_from_home = number_at(body, "\"km_from_home\"", termp);

    size_t colon = body.find(':', lastp);
    size_t value = body.find_first_not_of(" \n\r\t", colon + 1);
    if (body.substr(value, 4) == "null") {
        tx.has_last = false;
    } else {
        tx.has_last = true;
        tx.last_ts = string_at(body, "\"timestamp\"", value);
        tx.last_km = number_at(body, "\"km_from_current\"", value);
    }
    return tx;
}

inline int two(std::string_view s, size_t p) {
    return int(s[p] - '0') * 10 + int(s[p + 1] - '0');
}

inline int four(std::string_view s, size_t p) {
    return int(s[p] - '0') * 1000 + int(s[p + 1] - '0') * 100 + int(s[p + 2] - '0') * 10 + int(s[p + 3] - '0');
}

int weekday_monday0(int y, int m, int d) {
    static constexpr int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    if (m < 3) --y;
    int dow = (y + y / 4 - y / 100 + y / 400 + t[m - 1] + d) % 7;
    return (dow + 6) % 7;
}

int days_from_civil(int y, unsigned m, unsigned d) {
    y -= m <= 2;
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<int>(doe) - 719468;
}

int epoch_minutes(std::string_view ts) {
    return days_from_civil(four(ts, 0), unsigned(two(ts, 5)), unsigned(two(ts, 8))) * 1440 + two(ts, 11) * 60 + two(ts, 14);
}

int16_t mcc_risk_q(std::string_view mcc) {
    if (mcc == "5411") return 1500;
    if (mcc == "5812") return 3000;
    if (mcc == "5912") return 2000;
    if (mcc == "5944") return 4500;
    if (mcc == "7801") return 8000;
    if (mcc == "7802") return 7500;
    if (mcc == "7995") return 8500;
    if (mcc == "4511") return 3500;
    if (mcc == "5311") return 2500;
    if (mcc == "5999") return 5000;
    return 5000;
}

void vectorize(const Tx& tx, int16_t out[silent::Dims]) {
    int y = four(tx.requested_at, 0);
    int m = two(tx.requested_at, 5);
    int d = two(tx.requested_at, 8);
    int h = two(tx.requested_at, 11);
    out[0] = silent::qclamp01(tx.amount / 10000.0);
    out[1] = silent::qclamp01(double(tx.installments) / 12.0);
    out[2] = silent::qclamp01((tx.amount / tx.customer_avg) / 10.0);
    out[3] = silent::qclamp01(double(h) / 23.0);
    out[4] = silent::qclamp01(double(weekday_monday0(y, m, d)) / 6.0);
    if (tx.has_last) {
        out[5] = silent::qclamp01(double(epoch_minutes(tx.requested_at) - epoch_minutes(tx.last_ts)) / 1440.0);
        out[6] = silent::qclamp01(tx.last_km / 1000.0);
    } else {
        out[5] = -10000;
        out[6] = -10000;
    }
    out[7] = silent::qclamp01(tx.km_from_home / 1000.0);
    out[8] = silent::qclamp01(double(tx.tx_count_24h) / 20.0);
    out[9] = tx.is_online ? 10000 : 0;
    out[10] = tx.card_present ? 10000 : 0;
    out[11] = tx.known_merchants.find(tx.merchant_id) == std::string_view::npos ? 10000 : 0;
    out[12] = mcc_risk_q(tx.mcc);
    out[13] = silent::qclamp01(tx.merchant_avg / 10000.0);
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) profile_fatal("cannot open input file");
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
    profile_fatal("unterminated object");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: profile index.bin test-data.json [nprobe...]\n";
        return 2;
    }
    std::vector<int> probes;
    for (int i = 3; i < argc; ++i) probes.push_back(std::atoi(argv[i]));
    if (probes.empty()) probes = {4, 8, 12, 16, 24, 32};

    silent::MappedIndex index(argv[1]);
    int repair_min = std::getenv("REPAIR_MIN") ? std::atoi(std::getenv("REPAIR_MIN")) : 2;
    int repair_max = std::getenv("REPAIR_MAX") ? std::atoi(std::getenv("REPAIR_MAX")) : 3;
    std::string json = read_file(argv[2]);
    std::string_view all(json);

    std::vector<std::array<int16_t, silent::Dims>> queries;
    std::vector<bool> expected;
    queries.reserve(60000);
    expected.reserve(60000);

    auto parse_t0 = Clock::now();
    size_t p = 0;
    while ((p = all.find("\"request\"", p)) != std::string_view::npos) {
        size_t open = all.find('{', p);
        size_t end = object_end(all, open);
        std::string_view req = all.substr(open, end - open);
        auto tx = parse_tx(req);
        std::array<int16_t, silent::Dims> q{};
        vectorize(tx, q.data());
        queries.push_back(q);

        size_t exp = all.find("\"expected_approved\"", end);
        size_t colon = all.find(':', exp);
        size_t val = all.find_first_not_of(" \n\r\t", colon + 1);
        expected.push_back(all.substr(val, 4) == "true");
        p = end;
    }
    auto parse_ms = std::chrono::duration<double, std::milli>(Clock::now() - parse_t0).count();
    std::cout << "parsed/vectorized " << queries.size() << " payloads in " << parse_ms
              << "ms (" << (parse_ms * 1000.0 / queries.size()) << "us each)\n";

    for (int nprobe : probes) {
        uint32_t fp = 0, fn = 0;
        uint32_t repairs = 0;
        uint64_t initial_scanned = 0;
        uint64_t initial_pruned = 0;
        uint64_t repair_scanned = 0;
        uint64_t repair_pruned = 0;
        std::array<uint32_t, 6> initial_frauds{};
        std::array<uint32_t, 6> final_frauds{};
        std::array<std::array<uint32_t, 6>, 6> fraud_transitions{};
        std::array<uint32_t, 6> ok_by_initial{};
        std::array<uint32_t, 6> fail_by_initial{};
        std::array<uint64_t, 6> ok_min_worst{};
        std::array<uint64_t, 6> ok_max_worst{};
        std::array<uint64_t, 6> fail_min_worst{};
        std::array<uint64_t, 6> fail_max_worst{};
        ok_min_worst.fill(UINT64_MAX);
        fail_min_worst.fill(UINT64_MAX);
        std::vector<double> per_query_us;
        per_query_us.reserve(queries.size());
        auto t0 = Clock::now();
        for (size_t i = 0; i < queries.size(); ++i) {
            auto q0 = Clock::now();
            silent::SearchStats stats;
            uint8_t fraud = index.search(queries[i].data(), nprobe, repair_min, repair_max, &stats);
            per_query_us.push_back(std::chrono::duration<double, std::micro>(Clock::now() - q0).count());
            repairs += stats.repair_triggered ? 1U : 0U;
            initial_scanned += stats.initial_scanned;
            initial_pruned += stats.initial_pruned;
            repair_scanned += stats.repair_scanned;
            repair_pruned += stats.repair_pruned;
            ++initial_frauds[std::min<uint8_t>(stats.initial_fraud, 5)];
            ++final_frauds[std::min<uint8_t>(stats.final_fraud, 5)];
            ++fraud_transitions[std::min<uint8_t>(stats.initial_fraud, 5)][std::min<uint8_t>(stats.final_fraud, 5)];
            bool approved = fraud < 3;
            uint8_t initial = std::min<uint8_t>(stats.initial_fraud, 5);
            if (approved != expected[i]) {
                ++fail_by_initial[initial];
                fail_min_worst[initial] = std::min<uint64_t>(fail_min_worst[initial], stats.initial_worst_dist);
                fail_max_worst[initial] = std::max<uint64_t>(fail_max_worst[initial], stats.initial_worst_dist);
                if (approved) ++fn;
                else ++fp;
            } else {
                ++ok_by_initial[initial];
                ok_min_worst[initial] = std::min<uint64_t>(ok_min_worst[initial], stats.initial_worst_dist);
                ok_max_worst[initial] = std::max<uint64_t>(ok_max_worst[initial], stats.initial_worst_dist);
            }
        }
        auto ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        std::sort(per_query_us.begin(), per_query_us.end());
        auto pct = [&](double p) {
            size_t idx = std::min(per_query_us.size() - 1, static_cast<size_t>(p * double(per_query_us.size() - 1)));
            return per_query_us[idx];
        };
        std::cout << "nprobe=" << nprobe
                  << " search_ms=" << ms
                  << " avg_us=" << (ms * 1000.0 / queries.size())
                  << " p50_us=" << pct(0.50)
                  << " p95_us=" << pct(0.95)
                  << " p99_us=" << pct(0.99)
                  << " max_us=" << per_query_us.back()
                  << " repairs=" << repairs
                  << " repair_rate=" << (100.0 * double(repairs) / queries.size())
                  << "%"
                  << " avg_initial_scanned=" << (double(initial_scanned) / queries.size())
                  << " avg_initial_pruned=" << (double(initial_pruned) / queries.size())
                  << " avg_repair_scanned=" << (double(repair_scanned) / queries.size())
                  << " avg_repair_pruned=" << (double(repair_pruned) / queries.size())
                  << " initial_frauds=["
                  << initial_frauds[0] << "," << initial_frauds[1] << "," << initial_frauds[2] << ","
                  << initial_frauds[3] << "," << initial_frauds[4] << "," << initial_frauds[5] << "]"
                  << " final_frauds=["
                  << final_frauds[0] << "," << final_frauds[1] << "," << final_frauds[2] << ","
                  << final_frauds[3] << "," << final_frauds[4] << "," << final_frauds[5] << "]"
                  << " transitions=[";
        for (size_t from = 0; from < fraud_transitions.size(); ++from) {
            if (from != 0) std::cout << ";";
            std::cout << fraud_transitions[from][0] << "," << fraud_transitions[from][1] << ","
                      << fraud_transitions[from][2] << "," << fraud_transitions[from][3] << ","
                      << fraud_transitions[from][4] << "," << fraud_transitions[from][5];
        }
        std::cout << "]"
                  << " worst_ranges=[";
        for (size_t f = 0; f < 6; ++f) {
            if (f != 0) std::cout << ";";
            std::cout << ok_by_initial[f] << ":"
                      << (ok_by_initial[f] ? ok_min_worst[f] : 0) << ":"
                      << (ok_by_initial[f] ? ok_max_worst[f] : 0) << ":"
                      << fail_by_initial[f] << ":"
                      << (fail_by_initial[f] ? fail_min_worst[f] : 0) << ":"
                      << (fail_by_initial[f] ? fail_max_worst[f] : 0);
        }
        std::cout << "]"
                  << " fp=" << fp
                  << " fn=" << fn
                  << " failures=" << (fp + fn)
                  << " failure_rate=" << (100.0 * double(fp + fn) / queries.size())
                  << "%\n";
    }

    if (const char* full_env = std::getenv("ADAPTIVE_FULL_NPROBE")) {
        int full_nprobe = std::atoi(full_env);
        int adaptive_min = std::getenv("ADAPTIVE_MIN") ? std::atoi(std::getenv("ADAPTIVE_MIN")) : 1;
        int adaptive_max = std::getenv("ADAPTIVE_MAX") ? std::atoi(std::getenv("ADAPTIVE_MAX")) : 4;
        for (int fast_nprobe : probes) {
            uint32_t fp = 0, fn = 0;
            uint32_t reruns = 0;
            uint64_t fast_initial_scanned = 0;
            uint64_t full_initial_scanned = 0;
            uint64_t full_repair_scanned = 0;
            std::array<uint32_t, 6> fast_frauds{};
            std::array<uint32_t, 6> final_frauds{};
            std::vector<double> per_query_us;
            per_query_us.reserve(queries.size());
            auto t0 = Clock::now();
            for (size_t i = 0; i < queries.size(); ++i) {
                auto q0 = Clock::now();
                silent::SearchStats fast_stats;
                uint8_t fraud = index.search(queries[i].data(), fast_nprobe, 99, 0, &fast_stats);
                fast_initial_scanned += fast_stats.initial_scanned;
                ++fast_frauds[std::min<uint8_t>(fraud, 5)];
                if (fraud >= adaptive_min && fraud <= adaptive_max) {
                    ++reruns;
                    silent::SearchStats full_stats;
                    fraud = index.search(queries[i].data(), full_nprobe, repair_min, repair_max, &full_stats);
                    full_initial_scanned += full_stats.initial_scanned;
                    full_repair_scanned += full_stats.repair_scanned;
                }
                ++final_frauds[std::min<uint8_t>(fraud, 5)];
                per_query_us.push_back(std::chrono::duration<double, std::micro>(Clock::now() - q0).count());
                bool approved = fraud < 3;
                if (approved != expected[i]) {
                    if (approved) ++fn;
                    else ++fp;
                }
            }
            auto ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            std::sort(per_query_us.begin(), per_query_us.end());
            auto pct = [&](double p) {
                size_t idx = std::min(per_query_us.size() - 1, static_cast<size_t>(p * double(per_query_us.size() - 1)));
                return per_query_us[idx];
            };
            std::cout << "adaptive_fast=" << fast_nprobe
                      << " full=" << full_nprobe
                      << " trigger=" << adaptive_min << ".." << adaptive_max
                      << " search_ms=" << ms
                      << " avg_us=" << (ms * 1000.0 / queries.size())
                      << " p50_us=" << pct(0.50)
                      << " p95_us=" << pct(0.95)
                      << " p99_us=" << pct(0.99)
                      << " max_us=" << per_query_us.back()
                      << " reruns=" << reruns
                      << " rerun_rate=" << (100.0 * double(reruns) / queries.size())
                      << "%"
                      << " avg_fast_initial_scanned=" << (double(fast_initial_scanned) / queries.size())
                      << " avg_full_initial_scanned=" << (double(full_initial_scanned) / queries.size())
                      << " avg_full_repair_scanned=" << (double(full_repair_scanned) / queries.size())
                      << " fast_frauds=["
                      << fast_frauds[0] << "," << fast_frauds[1] << "," << fast_frauds[2] << ","
                      << fast_frauds[3] << "," << fast_frauds[4] << "," << fast_frauds[5] << "]"
                      << " final_frauds=["
                      << final_frauds[0] << "," << final_frauds[1] << "," << final_frauds[2] << ","
                      << final_frauds[3] << "," << final_frauds[4] << "," << final_frauds[5] << "]"
                      << " fp=" << fp
                      << " fn=" << fn
                      << " failures=" << (fp + fn)
                      << " failure_rate=" << (100.0 * double(fp + fn) / queries.size())
                      << "%\n";
        }
    }
    return 0;
}
