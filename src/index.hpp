#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace silent {

[[noreturn]] inline void index_fatal() { std::_Exit(1); }

constexpr int Dims = 14;
constexpr int Block = 8;
constexpr uint64_t Magic = 0x3149564936324852ULL; // "RH26IVI1", little-endian
constexpr uint32_t Version = 2;
constexpr uint32_t MaxRepairClusters = 2048;
constexpr uint32_t MaxRepairWords = (MaxRepairClusters + 63) / 64;
constexpr int DimPairs = Dims / 2;

struct IndexHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t n;
    uint32_t k;
    uint32_t total_blocks;
    uint32_t block_size;
    uint32_t dims;
    uint32_t reserved[8];
};

static_assert(sizeof(IndexHeader) == 64);

inline size_t align_up(size_t v, size_t a) {
    return (v + a - 1) & ~(a - 1);
}

struct IndexLayout {
    size_t centroids;
    size_t bbox_min;
    size_t bbox_max;
    size_t offsets;
    size_t counts;
    size_t labels;
    size_t blocks;
    size_t total;
};

inline IndexLayout layout_for(uint32_t k, uint32_t total_blocks) {
    IndexLayout l{};
    size_t off = sizeof(IndexHeader);
    l.centroids = off;
    off += size_t(k) * Dims * sizeof(int16_t);
    l.bbox_min = off;
    off += size_t(k) * Dims * sizeof(int16_t);
    l.bbox_max = off;
    off += size_t(k) * Dims * sizeof(int16_t);
    off = align_up(off, alignof(uint32_t));
    l.offsets = off;
    off += size_t(k + 1) * sizeof(uint32_t);
    l.counts = off;
    off += size_t(k) * sizeof(uint32_t);
    l.labels = off;
    off += size_t(total_blocks) * Block;
    off = align_up(off, alignof(int16_t));
    l.blocks = off;
    off += size_t(total_blocks) * Dims * Block * sizeof(int16_t);
    l.total = off;
    return l;
}

inline int16_t qround(double v) {
    if (v < -1.0) v = -1.0;
    if (v > 1.0) v = 1.0;
    return static_cast<int16_t>(__builtin_llround(v * 10000.0));
}

inline int16_t qclamp01(double v) {
    if (v < 0.0) v = 0.0;
    if (v > 1.0) v = 1.0;
    return static_cast<int16_t>(v * 10000.0 + 0.5);
}

inline uint32_t hsum256_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return uint32_t(_mm_cvtsi128_si32(sum));
}

inline uint32_t hsum128_epi32(__m128i v) {
    v = _mm_hadd_epi32(v, v);
    v = _mm_hadd_epi32(v, v);
    return uint32_t(_mm_cvtsi128_si32(v));
}

inline uint64_t dist_q16(const int16_t* a, const int16_t* b) {
    __m128i a16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
    __m128i b16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
    __m128i diff16 = _mm_sub_epi16(a16, b16);
    __m128i pair_sq = _mm_madd_epi16(diff16, diff16);
    pair_sq = _mm_hadd_epi32(pair_sq, pair_sq);
    pair_sq = _mm_hadd_epi32(pair_sq, pair_sq);
    uint64_t s = uint32_t(_mm_cvtsi128_si32(pair_sq));
    for (int d = 8; d < Dims; ++d) {
        int64_t diff = int64_t(a[d]) - int64_t(b[d]);
        s += uint64_t(diff * diff);
    }
    return s;
}

inline size_t block_pair_offset(int dim_pair, uint32_t lane) {
    return size_t(dim_pair) * Block * 2 + size_t(lane) * 2;
}

inline uint64_t dist_centroid(const int16_t* q, const int16_t* centroids, uint32_t c) {
    return dist_q16(q, centroids + size_t(c) * Dims);
}

inline uint64_t bbox_lower_bound_raw(const int16_t* q, const int16_t* mn, const int16_t* mx) {
    __m128i qv = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q));
    __m128i mnv = _mm_loadu_si128(reinterpret_cast<const __m128i*>(mn));
    __m128i mxv = _mm_loadu_si128(reinterpret_cast<const __m128i*>(mx));
    __m128i zero = _mm_setzero_si128();
    __m128i below = _mm_sub_epi16(mnv, qv);
    __m128i above = _mm_sub_epi16(qv, mxv);
    __m128i diff = _mm_max_epi16(_mm_max_epi16(below, above), zero);
    uint64_t s = hsum128_epi32(_mm_madd_epi16(diff, diff));
    for (int d = 8; d < Dims; ++d) {
        int64_t diff = 0;
        if (q[d] < mn[d]) diff = int64_t(mn[d]) - q[d];
        else if (q[d] > mx[d]) diff = int64_t(q[d]) - mx[d];
        s += uint64_t(diff * diff);
    }
    return s;
}

inline uint64_t bbox_lower_bound(const int16_t* q, const int16_t* bmin, const int16_t* bmax, uint32_t c) {
    return bbox_lower_bound_raw(q, bmin + size_t(c) * Dims, bmax + size_t(c) * Dims);
}

struct SearchStats {
    uint32_t initial_scanned = 0;
    uint32_t initial_pruned = 0;
    uint32_t initial_blocks = 0;
    uint32_t repair_scanned = 0;
    uint32_t repair_pruned = 0;
    uint32_t repair_blocks = 0;
    uint32_t repair_candidates = 0;
    bool repair_triggered = false;
    uint8_t initial_fraud = 0;
    uint8_t final_fraud = 0;
    uint64_t initial_worst_dist = 0;
    uint64_t final_worst_dist = 0;
    uint64_t centroid_cycles = 0;
    uint64_t initial_cycles = 0;
    uint64_t repair_select_cycles = 0;
    uint64_t repair_sort_cycles = 0;
    uint64_t repair_scan_cycles = 0;
};

inline uint64_t profile_cycles_now() {
#if defined(__x86_64__) || defined(__i386__)
    return __rdtsc();
#else
    return 0;
#endif
}

class MappedIndex {
public:
    explicit MappedIndex(const std::string& path) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) index_fatal();
        struct stat st {};
        if (fstat(fd, &st) != 0) {
            close(fd);
            index_fatal();
        }
        size_ = static_cast<size_t>(st.st_size);
        (void)posix_fadvise(fd, 0, static_cast<off_t>(size_), POSIX_FADV_WILLNEED);

        const char* mmap_env = std::getenv("INDEX_MMAP");
        mapped_ = mmap_env && (mmap_env[0] == '1' || std::strcmp(mmap_env, "true") == 0);
        if (mapped_) {
            raw_ = static_cast<uint8_t*>(mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd, 0));
            if (raw_ == MAP_FAILED) {
                close(fd);
                index_fatal();
            }
            (void)madvise(raw_, size_, MADV_WILLNEED);
            close(fd);
        } else {
            storage_.resize(size_);
            size_t off = 0;
            while (off < size_) {
                ssize_t n = ::read(fd, storage_.data() + off, size_ - off);
                if (n > 0) {
                    off += size_t(n);
                    continue;
                }
                if (n < 0 && errno == EINTR) continue;
                close(fd);
                index_fatal();
            }
            close(fd);
            raw_ = storage_.data();
        }

        header_ = reinterpret_cast<const IndexHeader*>(raw_);
        if (header_->magic != Magic || header_->version != Version || header_->dims != Dims || header_->block_size != Block) {
            index_fatal();
        }
        layout_ = layout_for(header_->k, header_->total_blocks);
        if (layout_.total > size_) index_fatal();

        centroids_ = reinterpret_cast<const int16_t*>(raw_ + layout_.centroids);
        bbox_min_ = reinterpret_cast<const int16_t*>(raw_ + layout_.bbox_min);
        bbox_max_ = reinterpret_cast<const int16_t*>(raw_ + layout_.bbox_max);
        offsets_ = reinterpret_cast<const uint32_t*>(raw_ + layout_.offsets);
        counts_ = reinterpret_cast<const uint32_t*>(raw_ + layout_.counts);
        labels_ = raw_ + layout_.labels;
        blocks_ = reinterpret_cast<const int16_t*>(raw_ + layout_.blocks);
        build_aux_soa();
        warm_pages();
    }

    MappedIndex(const MappedIndex&) = delete;
    MappedIndex& operator=(const MappedIndex&) = delete;

    ~MappedIndex() {
        if (mapped_ && raw_ && raw_ != MAP_FAILED) munmap(raw_, size_);
        raw_ = nullptr;
    }

    uint32_t k() const { return header_->k; }
    uint32_t n() const { return header_->n; }
    uint32_t total_blocks() const { return header_->total_blocks; }

    uint8_t search(const int16_t q[Dims], int nprobe, int repair_min = 2, int repair_max = 3, SearchStats* stats = nullptr) const {
        const uint32_t kk = k();
        if (nprobe < 1) nprobe = 1;
        if (nprobe > 64) nprobe = 64;
        if (uint32_t(nprobe) > kk) nprobe = int(kk);
        if (repair_min > repair_max && !centroids_soa_.empty()) {
            if (nprobe == 1 && kk >= 1) {
                return search_one_no_repair_avx2(q, stats);
            }
            if (nprobe == 2 && kk >= 2) {
                return search_two_no_repair_avx2(q, stats);
            }
            return search_n_no_repair_avx2(q, nprobe, stats);
        }
        if (nprobe == 2 && kk >= 2 && !centroids_soa_.empty()) {
            return search_two_with_repair_avx2(q, repair_min, repair_max, stats);
        }

        std::array<uint32_t, 64> best_c{};
        std::array<uint64_t, 64> best_d{};
        int used = 0;
        uint64_t worst = 0;
        int worst_i = 0;
        for (uint32_t c = 0; c < kk; ++c) {
            uint64_t d = dist_centroid(q, centroids_, c);
            if (used < nprobe) {
                best_c[used] = c;
                best_d[used] = d;
                if (used == 0 || d > worst) {
                    worst = d;
                    worst_i = used;
                }
                ++used;
            } else if (d < worst) {
                best_c[worst_i] = c;
                best_d[worst_i] = d;
                worst = best_d[0];
                worst_i = 0;
                for (int i = 1; i < used; ++i) {
                    if (best_d[i] > worst) {
                        worst = best_d[i];
                        worst_i = i;
                    }
                }
            }
        }

        for (int i = 1; i < used; ++i) {
            auto c = best_c[i];
            auto d = best_d[i];
            int j = i - 1;
            while (j >= 0 && best_d[j] > d) {
                best_d[j + 1] = best_d[j];
                best_c[j + 1] = best_c[j];
                --j;
            }
            best_d[j + 1] = d;
            best_c[j + 1] = c;
        }

        Top5 top;
        std::array<uint64_t, MaxRepairWords> scanned{};
        const uint32_t scanned_words = (kk + 63) / 64;
        if (scanned_words > scanned.size()) index_fatal();
        for (int i = 0; i < used; ++i) {
            uint32_t c = best_c[i];
            if (i > 0 && bbox_lower_bound(q, bbox_min_, bbox_max_, c) >= top.worst_dist()) {
                scanned[c >> 6] |= uint64_t(1) << (c & 63);
                if (stats) ++stats->initial_pruned;
                continue;
            }
            uint32_t blocks = scan_cluster(c, q, top, stats);
            if (stats) {
                ++stats->initial_scanned;
                stats->initial_blocks += blocks;
            }
            scanned[c >> 6] |= uint64_t(1) << (c & 63);
        }

        uint8_t fraud = top.fraud_count();
        if (stats) {
            stats->initial_fraud = fraud;
            stats->initial_worst_dist = top.worst_dist();
        }
        if (fraud >= repair_min && fraud <= repair_max) {
            if (stats) stats->repair_triggered = true;
            for (uint32_t c = 0; c < kk; ++c) {
                if (scanned[c >> 6] & (uint64_t(1) << (c & 63))) continue;
                if (bbox_lower_bound(q, bbox_min_, bbox_max_, c) >= top.worst_dist()) {
                    if (stats) ++stats->repair_pruned;
                    continue;
                }
                uint32_t blocks = scan_cluster(c, q, top, stats);
                if (stats) {
                    ++stats->repair_scanned;
                    stats->repair_blocks += blocks;
                }
            }
            fraud = top.fraud_count();
        }
        if (stats) {
            stats->final_fraud = fraud;
            stats->final_worst_dist = top.worst_dist();
        }
        return fraud;
    }

private:
    struct Top5 {
        std::array<uint32_t, 5> dist;
        std::array<uint8_t, 5> label;
        int worst;

        Top5() : dist{}, label{}, worst(0) {
            dist.fill(uint32_t(INT32_MAX));
            label.fill(0);
        }

        uint32_t worst_dist() const { return dist[worst]; }

        void add(uint32_t d, uint8_t l) {
            if (d >= dist[worst]) return;
            dist[worst] = d;
            label[worst] = l;
            worst = 0;
            for (int i = 1; i < 5; ++i) {
                if (dist[i] > dist[worst]) worst = i;
            }
        }

        uint8_t fraud_count() const {
            return uint8_t(label[0] + label[1] + label[2] + label[3] + label[4]);
        }
    };

    struct QueryPairs {
        alignas(32) __m256i pair[DimPairs];
    };

    static QueryPairs make_query_pairs(const int16_t q[Dims]) {
        QueryPairs out{};
        for (int pair = 0; pair < DimPairs; ++pair) {
            int d = pair * 2;
            uint32_t packed = uint32_t(uint16_t(q[d])) | (uint32_t(uint16_t(q[d + 1])) << 16);
            out.pair[pair] = _mm256_set1_epi32(int(packed));
        }
        return out;
    }

    void build_aux_soa() {
        const uint32_t kk = k();
        centroids_soa_.resize(size_t(kk) * Dims);
        centroids_pair_soa_.resize(size_t(DimPairs) * kk * 2);
        bbox_min_soa_.resize(size_t(kk) * Dims);
        bbox_max_soa_.resize(size_t(kk) * Dims);
        for (uint32_t c = 0; c < kk; ++c) {
            const int16_t* centroid = centroids_ + size_t(c) * Dims;
            const int16_t* bmin = bbox_min_ + size_t(c) * Dims;
            const int16_t* bmax = bbox_max_ + size_t(c) * Dims;
            for (int d = 0; d < Dims; ++d) {
                size_t off = size_t(d) * kk + c;
                centroids_soa_[off] = centroid[d];
                bbox_min_soa_[off] = bmin[d];
                bbox_max_soa_[off] = bmax[d];
            }
            for (int pair = 0; pair < DimPairs; ++pair) {
                int d = pair * 2;
                size_t off = size_t(pair) * kk * 2 + size_t(c) * 2;
                centroids_pair_soa_[off] = centroid[d];
                centroids_pair_soa_[off + 1] = centroid[d + 1];
            }
        }
    }

    void bbox_lower_bounds8_avx2(const int16_t q[Dims], uint32_t c, uint32_t out[8]) const {
        const uint32_t kk = k();
        __m256i acc = _mm256_setzero_si256();
        const __m128i zero = _mm_setzero_si128();
        for (int d = 0; d < Dims; d += 2) {
            const int16_t* mn0 = bbox_min_soa_.data() + size_t(d) * kk + c;
            const int16_t* mx0 = bbox_max_soa_.data() + size_t(d) * kk + c;
            const int16_t* mn1 = bbox_min_soa_.data() + size_t(d + 1) * kk + c;
            const int16_t* mx1 = bbox_max_soa_.data() + size_t(d + 1) * kk + c;

            __m128i q0 = _mm_set1_epi16(q[d]);
            __m128i q1 = _mm_set1_epi16(q[d + 1]);
            __m128i diff0 = _mm_max_epi16(
                _mm_max_epi16(_mm_sub_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(mn0)), q0),
                              _mm_sub_epi16(q0, _mm_loadu_si128(reinterpret_cast<const __m128i*>(mx0)))),
                zero);
            __m128i diff1 = _mm_max_epi16(
                _mm_max_epi16(_mm_sub_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(mn1)), q1),
                              _mm_sub_epi16(q1, _mm_loadu_si128(reinterpret_cast<const __m128i*>(mx1)))),
                zero);
            __m128i lo = _mm_unpacklo_epi16(diff0, diff1);
            __m128i hi = _mm_unpackhi_epi16(diff0, diff1);
            __m256i pairs = _mm256_set_m128i(hi, lo);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, pairs));
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), acc);
    }

    void find_two_centroids_avx2(const int16_t q[Dims], const QueryPairs& qp, uint32_t& best0, uint32_t& best1) const {
        const uint32_t kk = k();
        best0 = 0;
        best1 = 0;
        uint32_t dist0 = UINT32_MAX;
        uint32_t dist1 = UINT32_MAX;
        uint32_t c = 0;
        const __m256i lane_inc = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256i all_ones = _mm256_cmpeq_epi32(lane_inc, lane_inc);
        __m256i best_dist_v = _mm256_set1_epi32(INT32_MAX);
        __m256i best_idx_v = _mm256_setzero_si256();
        __m256i second_dist_v = _mm256_set1_epi32(INT32_MAX);
        __m256i second_idx_v = _mm256_setzero_si256();
        for (; c + 8 <= kk; c += 8) {
            __m256i acc = _mm256_setzero_si256();
            for (int pair = 0; pair < DimPairs; ++pair) {
                __m256i ref = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(centroids_pair_soa_.data() + size_t(pair) * kk * 2 + size_t(c) * 2));
                __m256i diff = _mm256_sub_epi16(qp.pair[pair], ref);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(diff, diff));
            }
            __m256i idx = _mm256_add_epi32(_mm256_set1_epi32(int(c)), lane_inc);
            __m256i old_best_dist = best_dist_v;
            __m256i old_best_idx = best_idx_v;
            __m256i better_best = _mm256_cmpgt_epi32(best_dist_v, acc);
            __m256i better_second = _mm256_and_si256(_mm256_xor_si256(better_best, all_ones),
                                                     _mm256_cmpgt_epi32(second_dist_v, acc));

            best_dist_v = _mm256_blendv_epi8(best_dist_v, acc, better_best);
            best_idx_v = _mm256_blendv_epi8(best_idx_v, idx, better_best);
            second_dist_v = _mm256_blendv_epi8(second_dist_v, old_best_dist, better_best);
            second_idx_v = _mm256_blendv_epi8(second_idx_v, old_best_idx, better_best);
            second_dist_v = _mm256_blendv_epi8(second_dist_v, acc, better_second);
            second_idx_v = _mm256_blendv_epi8(second_idx_v, idx, better_second);
        }
        alignas(32) uint32_t best_dist_lanes[8];
        alignas(32) uint32_t best_idx_lanes[8];
        alignas(32) uint32_t second_dist_lanes[8];
        alignas(32) uint32_t second_idx_lanes[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(best_dist_lanes), best_dist_v);
        _mm256_store_si256(reinterpret_cast<__m256i*>(best_idx_lanes), best_idx_v);
        _mm256_store_si256(reinterpret_cast<__m256i*>(second_dist_lanes), second_dist_v);
        _mm256_store_si256(reinterpret_cast<__m256i*>(second_idx_lanes), second_idx_v);
        auto add_centroid = [&](uint32_t d, uint32_t cluster) {
            if (d < dist0) {
                dist1 = dist0;
                best1 = best0;
                dist0 = d;
                best0 = cluster;
            } else if (d < dist1) {
                dist1 = d;
                best1 = cluster;
            }
        };
        for (uint32_t lane = 0; lane < 8; ++lane) add_centroid(best_dist_lanes[lane], best_idx_lanes[lane]);
        for (uint32_t lane = 0; lane < 8; ++lane) add_centroid(second_dist_lanes[lane], second_idx_lanes[lane]);
        for (; c < kk; ++c) {
            uint32_t d = static_cast<uint32_t>(dist_centroid(q, centroids_, c));
            add_centroid(d, c);
        }
    }

    uint8_t search_two_with_repair_avx2(const int16_t q[Dims], int repair_min, int repair_max, SearchStats* stats) const {
        const uint32_t kk = k();
        uint64_t phase_start = stats ? profile_cycles_now() : 0;
        QueryPairs qp = make_query_pairs(q);
        uint32_t best0 = 0;
        uint32_t best1 = 1;
        find_two_centroids_avx2(q, qp, best0, best1);
        if (stats) stats->centroid_cycles += profile_cycles_now() - phase_start;

        phase_start = stats ? profile_cycles_now() : 0;
        Top5 top;
        uint32_t blocks = scan_cluster(best0, q, qp, top, stats);
        if (stats) {
            ++stats->initial_scanned;
            stats->initial_blocks += blocks;
        }

        if (bbox_lower_bound(q, bbox_min_, bbox_max_, best1) >= top.worst_dist()) {
            if (stats) ++stats->initial_pruned;
        } else {
            blocks = scan_cluster(best1, q, qp, top, stats);
            if (stats) {
                ++stats->initial_scanned;
                stats->initial_blocks += blocks;
            }
        }
        if (stats) stats->initial_cycles += profile_cycles_now() - phase_start;

        uint8_t fraud = top.fraud_count();
        if (stats) {
            stats->initial_fraud = fraud;
            stats->initial_worst_dist = top.worst_dist();
        }

        if (fraud >= repair_min && fraud <= repair_max) {
            if (stats) stats->repair_triggered = true;
            static thread_local std::array<uint64_t, MaxRepairClusters> candidates;
            uint32_t candidate_count = 0;
            const uint64_t repair_limit = top.worst_dist();
            alignas(32) uint32_t bounds[8];
            uint32_t c = 0;
            phase_start = stats ? profile_cycles_now() : 0;
            for (; c + 8 <= kk; c += 8) {
                bbox_lower_bounds8_avx2(q, c, bounds);
                for (uint32_t lane = 0; lane < 8; ++lane) {
                    uint32_t cluster = c + lane;
                    if (cluster == best0 || cluster == best1) continue;
                    if (uint64_t(bounds[lane]) >= repair_limit) {
                        if (stats) ++stats->repair_pruned;
                        continue;
                    }
                    candidates[candidate_count++] = (uint64_t(bounds[lane]) << 32) | cluster;
                }
            }
            for (; c < kk; ++c) {
                if (c == best0 || c == best1) continue;
                uint64_t bound = bbox_lower_bound(q, bbox_min_, bbox_max_, c);
                if (bound >= repair_limit) {
                    if (stats) ++stats->repair_pruned;
                    continue;
                }
                candidates[candidate_count++] = (bound << 32) | c;
            }
            if (stats) stats->repair_select_cycles += profile_cycles_now() - phase_start;
            phase_start = stats ? profile_cycles_now() : 0;
            if (stats) stats->repair_candidates += candidate_count;
            std::sort(candidates.begin(), candidates.begin() + candidate_count);
            if (stats) stats->repair_sort_cycles += profile_cycles_now() - phase_start;
            phase_start = stats ? profile_cycles_now() : 0;
            for (uint32_t i = 0; i < candidate_count; ++i) {
                uint64_t packed = candidates[i];
                uint64_t bound = packed >> 32;
                if (bound >= top.worst_dist()) {
                    if (stats) ++stats->repair_pruned;
                    continue;
                }
                blocks = scan_cluster(uint32_t(packed), q, qp, top, stats);
                if (stats) {
                    ++stats->repair_scanned;
                    stats->repair_blocks += blocks;
                }
            }
            if (stats) stats->repair_scan_cycles += profile_cycles_now() - phase_start;
            fraud = top.fraud_count();
        }

        if (stats) {
            stats->final_fraud = fraud;
            stats->final_worst_dist = top.worst_dist();
        }
        return fraud;
    }

    uint8_t search_two_no_repair_avx2(const int16_t q[Dims], SearchStats* stats) const {
        QueryPairs qp = make_query_pairs(q);
        uint32_t best0 = 0;
        uint32_t best1 = 1;
        find_two_centroids_avx2(q, qp, best0, best1);

        Top5 top;
        uint32_t blocks = scan_cluster(best0, q, qp, top, stats);
        if (stats) {
            ++stats->initial_scanned;
            stats->initial_blocks += blocks;
        }
        if (bbox_lower_bound(q, bbox_min_, bbox_max_, best1) >= top.worst_dist()) {
            if (stats) ++stats->initial_pruned;
        } else {
            blocks = scan_cluster(best1, q, qp, top, stats);
            if (stats) {
                ++stats->initial_scanned;
                stats->initial_blocks += blocks;
            }
        }
        uint8_t fraud = top.fraud_count();
        if (stats) {
            stats->initial_fraud = fraud;
            stats->final_fraud = fraud;
            stats->initial_worst_dist = top.worst_dist();
            stats->final_worst_dist = top.worst_dist();
        }
        return fraud;
    }

    uint8_t search_one_no_repair_avx2(const int16_t q[Dims], SearchStats* stats) const {
        const uint32_t kk = k();
        uint32_t best = 0;
        uint32_t best_dist = UINT32_MAX;
        uint32_t c = 0;
        const __m256i lane_inc = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        __m256i best_dist_v = _mm256_set1_epi32(INT32_MAX);
        __m256i best_idx_v = _mm256_setzero_si256();
        for (; c + 8 <= kk; c += 8) {
            __m256i acc = _mm256_setzero_si256();
            for (int d = 0; d < Dims; d += 2) {
                const int16_t* row0 = centroids_soa_.data() + size_t(d) * kk + c;
                const int16_t* row1 = centroids_soa_.data() + size_t(d + 1) * kk + c;
                __m128i ref0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row0));
                __m128i ref1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row1));
                __m128i diff0 = _mm_sub_epi16(_mm_set1_epi16(q[d]), ref0);
                __m128i diff1 = _mm_sub_epi16(_mm_set1_epi16(q[d + 1]), ref1);
                __m128i lo = _mm_unpacklo_epi16(diff0, diff1);
                __m128i hi = _mm_unpackhi_epi16(diff0, diff1);
                __m256i pairs = _mm256_set_m128i(hi, lo);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, pairs));
            }
            __m256i idx = _mm256_add_epi32(_mm256_set1_epi32(int(c)), lane_inc);
            __m256i better = _mm256_cmpgt_epi32(best_dist_v, acc);
            best_dist_v = _mm256_blendv_epi8(best_dist_v, acc, better);
            best_idx_v = _mm256_blendv_epi8(best_idx_v, idx, better);
        }
        alignas(32) uint32_t best_dist_lanes[8];
        alignas(32) uint32_t best_idx_lanes[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(best_dist_lanes), best_dist_v);
        _mm256_store_si256(reinterpret_cast<__m256i*>(best_idx_lanes), best_idx_v);
        for (uint32_t lane = 0; lane < 8; ++lane) {
            uint32_t d = best_dist_lanes[lane];
            if (d < best_dist) {
                best_dist = d;
                best = best_idx_lanes[lane];
            }
        }
        for (; c < kk; ++c) {
            uint32_t d = static_cast<uint32_t>(dist_centroid(q, centroids_, c));
            if (d < best_dist) {
                best_dist = d;
                best = c;
            }
        }

        Top5 top;
        uint32_t blocks = scan_cluster(best, q, top, stats);
        uint8_t fraud = top.fraud_count();
        if (stats) {
            stats->initial_scanned = 1;
            stats->initial_blocks = blocks;
            stats->initial_fraud = fraud;
            stats->final_fraud = fraud;
            stats->initial_worst_dist = top.worst_dist();
            stats->final_worst_dist = top.worst_dist();
        }
        return fraud;
    }

    uint8_t search_n_no_repair_avx2(const int16_t q[Dims], int nprobe, SearchStats* stats) const {
        const uint32_t kk = k();
        std::array<uint32_t, 64> best_c{};
        std::array<uint64_t, 64> best_d{};
        int used = 0;
        uint64_t worst = 0;
        int worst_i = 0;
        uint32_t c = 0;
        alignas(32) uint32_t dist[8];
        for (; c + 8 <= kk; c += 8) {
            __m256i acc = _mm256_setzero_si256();
            for (int d = 0; d < Dims; d += 2) {
                const int16_t* row0 = centroids_soa_.data() + size_t(d) * kk + c;
                const int16_t* row1 = centroids_soa_.data() + size_t(d + 1) * kk + c;
                __m128i ref0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row0));
                __m128i ref1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row1));
                __m128i diff0 = _mm_sub_epi16(_mm_set1_epi16(q[d]), ref0);
                __m128i diff1 = _mm_sub_epi16(_mm_set1_epi16(q[d + 1]), ref1);
                __m128i lo = _mm_unpacklo_epi16(diff0, diff1);
                __m128i hi = _mm_unpackhi_epi16(diff0, diff1);
                __m256i pairs = _mm256_set_m128i(hi, lo);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, pairs));
            }
            _mm256_store_si256(reinterpret_cast<__m256i*>(dist), acc);
            for (uint32_t lane = 0; lane < 8; ++lane) {
                uint64_t d = dist[lane];
                uint32_t cluster = c + lane;
                if (used < nprobe) {
                    best_c[used] = cluster;
                    best_d[used] = d;
                    if (used == 0 || d > worst) {
                        worst = d;
                        worst_i = used;
                    }
                    ++used;
                } else if (d < worst) {
                    best_c[worst_i] = cluster;
                    best_d[worst_i] = d;
                    worst = best_d[0];
                    worst_i = 0;
                    for (int i = 1; i < used; ++i) {
                        if (best_d[i] > worst) {
                            worst = best_d[i];
                            worst_i = i;
                        }
                    }
                }
            }
        }
        for (; c < kk; ++c) {
            uint64_t d = dist_centroid(q, centroids_, c);
            if (used < nprobe) {
                best_c[used] = c;
                best_d[used] = d;
                if (used == 0 || d > worst) {
                    worst = d;
                    worst_i = used;
                }
                ++used;
            } else if (d < worst) {
                best_c[worst_i] = c;
                best_d[worst_i] = d;
                worst = best_d[0];
                worst_i = 0;
                for (int i = 1; i < used; ++i) {
                    if (best_d[i] > worst) {
                        worst = best_d[i];
                        worst_i = i;
                    }
                }
            }
        }

        for (int i = 1; i < used; ++i) {
            auto cluster = best_c[i];
            auto d = best_d[i];
            int j = i - 1;
            while (j >= 0 && best_d[j] > d) {
                best_d[j + 1] = best_d[j];
                best_c[j + 1] = best_c[j];
                --j;
            }
            best_d[j + 1] = d;
            best_c[j + 1] = cluster;
        }

        Top5 top;
        for (int i = 0; i < used; ++i) {
            uint32_t cluster = best_c[i];
            if (i > 0 && bbox_lower_bound(q, bbox_min_, bbox_max_, cluster) >= top.worst_dist()) {
                if (stats) ++stats->initial_pruned;
                continue;
            }
            uint32_t blocks = scan_cluster(cluster, q, top, stats);
            if (stats) {
                ++stats->initial_scanned;
                stats->initial_blocks += blocks;
            }
        }
        uint8_t fraud = top.fraud_count();
        if (stats) {
            stats->initial_fraud = fraud;
            stats->final_fraud = fraud;
            stats->initial_worst_dist = top.worst_dist();
            stats->final_worst_dist = top.worst_dist();
        }
        return fraud;
    }

    uint32_t scan_cluster(uint32_t c, const int16_t q[Dims], Top5& top, SearchStats* stats = nullptr) const {
        QueryPairs qp = make_query_pairs(q);
        return scan_cluster(c, q, qp, top, stats);
    }

    uint32_t scan_cluster(uint32_t c, const int16_t q[Dims], const QueryPairs& qp, Top5& top, SearchStats* stats = nullptr) const {
        uint32_t start = offsets_[c];
        uint32_t count = counts_[c];
        uint32_t blocks = (count + Block - 1) / Block;
        for (uint32_t bi = 0; bi < blocks; ++bi) {
            uint32_t block_id = start + bi;
            if (bi + 1 < blocks) {
                uint32_t next_block_id = start + bi + 1;
                _mm_prefetch(reinterpret_cast<const char*>(blocks_ + size_t(next_block_id) * Dims * Block), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(labels_ + size_t(next_block_id) * Block), _MM_HINT_T0);
            }
            uint32_t valid = std::min<uint32_t>(Block, count - bi * Block);
            const int16_t* block = blocks_ + size_t(block_id) * Dims * Block;
            const uint8_t* labels = labels_ + size_t(block_id) * Block;
            if (valid == Block) {
                scan_block8_avx2(block, labels, qp, top);
                continue;
            }
            for (uint32_t lane = 0; lane < valid; ++lane) {
                uint32_t s = 0;
                uint32_t limit = top.worst_dist();
                for (int d = 0; d < Dims; ++d) {
                    int32_t diff = int32_t(q[d]) -
                                   int32_t(block[block_pair_offset(d / 2, lane) + size_t(d & 1)]);
                    s += uint32_t(diff * diff);
                    if (s >= limit) break;
                }
                top.add(s, labels[lane]);
            }
        }
        return blocks;
    }

    static void scan_block8_avx2(const int16_t* block, const uint8_t* labels, const QueryPairs& qp, Top5& top) {
        __m256i acc = _mm256_setzero_si256();
        for (int pair = 0; pair < DimPairs; ++pair) {
            __m256i ref = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(block + size_t(pair) * Block * 2));
            __m256i diff = _mm256_sub_epi16(qp.pair[pair], ref);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(diff, diff));
        }

        uint32_t limit = top.worst_dist();
        __m256i limit_v = _mm256_set1_epi32(int(limit));
        __m256i lt = _mm256_cmpgt_epi32(limit_v, acc);
        uint32_t mask = uint32_t(_mm256_movemask_ps(_mm256_castsi256_ps(lt)));
        if (mask == 0) return;

        alignas(32) uint32_t dist[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(dist), acc);
        while (mask != 0) {
            uint32_t lane = uint32_t(__builtin_ctz(mask));
            mask &= mask - 1;
            top.add(dist[lane], labels[lane]);
        }
    }

    void warm_pages() {
        long page = sysconf(_SC_PAGESIZE);
        if (page <= 0) page = 4096;
        volatile uint8_t sink = 0;
        for (size_t off = 0; off < size_; off += size_t(page)) {
            sink ^= raw_[off];
        }
        if (size_ > 0) sink ^= raw_[size_ - 1];
        warmup_ = sink;
    }

    size_t size_ = 0;
    std::vector<uint8_t> storage_;
    bool mapped_ = false;
    uint8_t* raw_ = nullptr;
    uint8_t warmup_ = 0;
    const IndexHeader* header_ = nullptr;
    IndexLayout layout_{};
    const int16_t* centroids_ = nullptr;
    const int16_t* bbox_min_ = nullptr;
    const int16_t* bbox_max_ = nullptr;
    const uint32_t* offsets_ = nullptr;
    const uint32_t* counts_ = nullptr;
    const uint8_t* labels_ = nullptr;
    const int16_t* blocks_ = nullptr;
    std::vector<int16_t> centroids_soa_;
    std::vector<int16_t> centroids_pair_soa_;
    std::vector<int16_t> bbox_min_soa_;
    std::vector<int16_t> bbox_max_soa_;
};

} // namespace silent
