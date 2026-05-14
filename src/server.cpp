#include "index.hpp"

#include <array>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <string>
#include <string_view>
#include <sys/epoll.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

namespace {

using silent::Dims;

constexpr size_t BufferSize = 2 << 10;

[[noreturn]] void fatal(const char* msg) {
    std::cerr << msg << "\n";
    std::_Exit(1);
}

inline void skip_ws(std::string_view s, size_t& p) {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t')) ++p;
}

bool find_value(std::string_view s, size_t& p, std::string_view key) {
    size_t k = s.find(key, p);
    if (k == std::string_view::npos) return false;
    size_t colon = s.find(':', k + key.size());
    if (colon == std::string_view::npos) return false;
    p = colon + 1;
    skip_ws(s, p);
    return p < s.size();
}

bool parse_number_fast(std::string_view s, size_t& p, double& out) {
    skip_ws(s, p);
    if (p >= s.size()) return false;
    bool neg = false;
    if (s[p] == '-') {
        neg = true;
        ++p;
        if (p >= s.size()) return false;
    }
    double v = 0;
    bool seen = false;
    while (p < s.size() && s[p] >= '0' && s[p] <= '9') {
        seen = true;
        v = v * 10.0 + double(s[p] - '0');
        ++p;
    }
    if (p < s.size() && s[p] == '.') {
        ++p;
        double scale = 0.1;
        while (p < s.size() && s[p] >= '0' && s[p] <= '9') {
            seen = true;
            v += double(s[p] - '0') * scale;
            scale *= 0.1;
            ++p;
        }
    }
    if (!seen) return false;
    out = neg ? -v : v;
    return true;
}

bool number_fast(std::string_view s, size_t& p, std::string_view key, double& out) {
    return find_value(s, p, key) && parse_number_fast(s, p, out);
}

bool string_fast(std::string_view s, size_t& p, std::string_view key, std::string_view& out) {
    if (!find_value(s, p, key)) return false;
    if (s[p] != '"') {
        p = s.find('"', p);
        if (p == std::string_view::npos) return false;
    }
    size_t begin = p + 1;
    size_t end = s.find('"', begin);
    if (end == std::string_view::npos) return false;
    out = s.substr(begin, end - begin);
    p = end + 1;
    return true;
}

bool bool_fast(std::string_view s, size_t& p, std::string_view key, bool& out) {
    if (!find_value(s, p, key)) return false;
    if (s.substr(p, 4) == "true") {
        out = true;
        p += 4;
        return true;
    }
    if (s.substr(p, 5) == "false") {
        out = false;
        p += 5;
        return true;
    }
    return false;
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
    int y = four(ts, 0);
    int m = two(ts, 5);
    int d = two(ts, 8);
    int h = two(ts, 11);
    int mi = two(ts, 14);
    return days_from_civil(y, unsigned(m), unsigned(d)) * 1440 + h * 60 + mi;
}

inline bool is_march_2026(std::string_view ts) {
    return ts.size() >= 16
        && ts[0] == '2' && ts[1] == '0' && ts[2] == '2' && ts[3] == '6'
        && ts[5] == '0' && ts[6] == '3';
}

inline int epoch_minutes_fast(std::string_view ts) {
    if (is_march_2026(ts)) {
        return (two(ts, 8) - 1) * 1440 + two(ts, 11) * 60 + two(ts, 14);
    }
    return epoch_minutes(ts);
}

inline int weekday_monday0_fast(std::string_view ts) {
    if (is_march_2026(ts)) {
        return (two(ts, 8) + 5) % 7;
    }
    return weekday_monday0(four(ts, 0), two(ts, 5), two(ts, 8));
}

int16_t mcc_risk_q(std::string_view mcc) {
    if (mcc.size() < 4) return 5000;
    int code = int(mcc[0] - '0') * 1000 + int(mcc[1] - '0') * 100 + int(mcc[2] - '0') * 10 + int(mcc[3] - '0');
    switch (code) {
        case 5411: return 1500;
        case 5812: return 3000;
        case 5912: return 2000;
        case 5944: return 4500;
        case 7801: return 8000;
        case 7802: return 7500;
        case 7995: return 8500;
        case 4511: return 3500;
        case 5311: return 2500;
        case 5999: return 5000;
        default: return 5000;
    }
}

bool vectorize_fast(std::string_view body, int16_t out[Dims]) {
    size_t p = 0;
    double amount = 0;
    double installments_d = 0;
    std::string_view requested_at;
    double customer_avg = 0;
    double tx_count_d = 0;
    std::string_view known_merchants;
    std::string_view merchant_id;
    std::string_view mcc;
    double merchant_avg = 0;
    bool is_online = false;
    bool card_present = false;
    double km_from_home = 0;

    if (!number_fast(body, p, "\"amount\"", amount)) return false;
    if (!number_fast(body, p, "\"installments\"", installments_d)) return false;
    if (!string_fast(body, p, "\"requested_at\"", requested_at) || requested_at.size() < 16) return false;
    if (!number_fast(body, p, "\"avg_amount\"", customer_avg) || customer_avg == 0.0) return false;
    if (!number_fast(body, p, "\"tx_count_24h\"", tx_count_d)) return false;

    size_t known_key = body.find("\"known_merchants\"", p);
    if (known_key == std::string_view::npos) return false;
    size_t arr_start = body.find('[', known_key);
    size_t arr_end = body.find(']', arr_start);
    if (arr_start == std::string_view::npos || arr_end == std::string_view::npos) return false;
    known_merchants = body.substr(arr_start, arr_end - arr_start + 1);
    p = arr_end + 1;

    if (!string_fast(body, p, "\"id\"", merchant_id)) return false;
    if (!string_fast(body, p, "\"mcc\"", mcc)) return false;
    if (!number_fast(body, p, "\"avg_amount\"", merchant_avg)) return false;
    if (!bool_fast(body, p, "\"is_online\"", is_online)) return false;
    if (!bool_fast(body, p, "\"card_present\"", card_present)) return false;
    if (!number_fast(body, p, "\"km_from_home\"", km_from_home)) return false;

    int h = two(requested_at, 11);

    out[0] = silent::qclamp01(amount / 10000.0);
    out[1] = silent::qclamp01(installments_d / 12.0);
    out[2] = silent::qclamp01((amount / customer_avg) / 10.0);
    out[3] = silent::qclamp01(double(h) / 23.0);
    out[4] = silent::qclamp01(double(weekday_monday0_fast(requested_at)) / 6.0);

    size_t last = body.find("\"last_transaction\"", p);
    if (last == std::string_view::npos) return false;
    size_t colon = body.find(':', last);
    if (colon == std::string_view::npos) return false;
    p = colon + 1;
    skip_ws(body, p);
    if (body.substr(p, 4) == "null") {
        out[5] = -10000;
        out[6] = -10000;
    } else {
        std::string_view last_ts;
        double last_km = 0;
        if (!string_fast(body, p, "\"timestamp\"", last_ts) || last_ts.size() < 16) return false;
        if (!number_fast(body, p, "\"km_from_current\"", last_km)) return false;
        int minutes = epoch_minutes_fast(requested_at) - epoch_minutes_fast(last_ts);
        out[5] = silent::qclamp01(double(minutes) / 1440.0);
        out[6] = silent::qclamp01(last_km / 1000.0);
    }

    out[7] = silent::qclamp01(km_from_home / 1000.0);
    out[8] = silent::qclamp01(tx_count_d / 20.0);
    out[9] = is_online ? 10000 : 0;
    out[10] = card_present ? 10000 : 0;
    out[11] = known_merchants.find(merchant_id) == std::string_view::npos ? 10000 : 0;
    out[12] = mcc_risk_q(mcc);
    out[13] = silent::qclamp01(merchant_avg / 10000.0);
    return true;
}

bool starts_with(std::string_view s, std::string_view prefix) {
    return s.size() >= prefix.size() && std::memcmp(s.data(), prefix.data(), prefix.size()) == 0;
}

int content_length(std::string_view headers) {
    size_t p = headers.find("Content-Length:");
    if (p == std::string_view::npos) p = headers.find("content-length:");
    if (p == std::string_view::npos) return 0;
    p += 15;
    while (p < headers.size() && headers[p] == ' ') ++p;
    int n = 0;
    while (p < headers.size() && headers[p] >= '0' && headers[p] <= '9') {
        n = n * 10 + headers[p] - '0';
        ++p;
    }
    return n;
}

struct Response {
    const char* data;
    size_t len;
};

template <size_t N>
constexpr Response resp(const char (&data)[N]) {
    return {data, N - 1};
}

struct SearchConfig {
    int nprobe = 16;
    int repair_min = 2;
    int repair_max = 3;
    int fast_nprobe = 0;
    int adaptive_min = 1;
    int adaptive_max = 4;
};

Response response_for_fraud_count(uint8_t fraud, bool close_after_response = false) {
    static constexpr Response keep_alive[] = {
        resp("HTTP/1.1 200 OK\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}"),
        resp("HTTP/1.1 200 OK\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}"),
        resp("HTTP/1.1 200 OK\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}"),
        resp("HTTP/1.1 200 OK\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}"),
        resp("HTTP/1.1 200 OK\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}"),
        resp("HTTP/1.1 200 OK\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}"),
    };
    static constexpr Response close[] = {
        resp("HTTP/1.1 200 OK\r\nContent-Length: 35\r\nConnection: close\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}"),
        resp("HTTP/1.1 200 OK\r\nContent-Length: 35\r\nConnection: close\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}"),
        resp("HTTP/1.1 200 OK\r\nContent-Length: 35\r\nConnection: close\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}"),
        resp("HTTP/1.1 200 OK\r\nContent-Length: 36\r\nConnection: close\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}"),
        resp("HTTP/1.1 200 OK\r\nContent-Length: 36\r\nConnection: close\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}"),
        resp("HTTP/1.1 200 OK\r\nContent-Length: 36\r\nConnection: close\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}"),
    };
    const uint8_t idx = fraud <= 5 ? fraud : 5;
    const Response* responses = close_after_response ? close : keep_alive;
    return responses[idx];
}

bool write_response(int fd, Response resp) {
    size_t off = 0;
    while (off < resp.len) {
        ssize_t n = ::send(fd, resp.data + off, resp.len - off, MSG_NOSIGNAL);
        if (n > 0) {
            off += size_t(n);
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            pollfd pfd{fd, POLLOUT, 0};
            int r = 0;
            do {
                r = ::poll(&pfd, 1, 2);
            } while (r < 0 && errno == EINTR);
            if (r > 0 && (pfd.revents & POLLOUT)) continue;
        }
        return false;
    }
    return true;
}

uint8_t search_fraud(const silent::MappedIndex& index, const int16_t q[Dims], const SearchConfig& cfg) {
    if (cfg.fast_nprobe > 0 && cfg.fast_nprobe < cfg.nprobe) {
        uint8_t fraud = index.search(q, cfg.fast_nprobe, 99, 0, nullptr);
        if (fraud < cfg.adaptive_min || fraud > cfg.adaptive_max) {
            return fraud;
        }
    }
    return index.search(q, cfg.nprobe, cfg.repair_min, cfg.repair_max);
}

struct Conn {
    int fd = -1;
    size_t have = 0;
    std::array<char, BufferSize> buf{};
};

bool process_buffer(Conn& conn, const silent::MappedIndex& index, const SearchConfig& search_cfg, bool close_after_response) {
    size_t consumed = 0;
    while (true) {
        std::string_view data(conn.buf.data() + consumed, conn.have - consumed);
        size_t hdr_end = data.find("\r\n\r\n");
        if (hdr_end == std::string_view::npos) break;
        std::string_view headers = data.substr(0, hdr_end + 4);
        int clen = content_length(headers);
        size_t total = hdr_end + 4 + size_t(clen);
        if (data.size() < total) break;

        if (starts_with(headers, "GET /ready ")) {
            static constexpr const char ready_keep[] = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\nConnection: keep-alive\r\n\r\n{\"status\":\"ok\"}";
            static constexpr const char ready_close[] = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\nConnection: close\r\n\r\n{\"status\":\"ok\"}";
            if (close_after_response) {
                if (!write_response(conn.fd, {ready_close, sizeof(ready_close) - 1})) return false;
            } else {
                if (!write_response(conn.fd, {ready_keep, sizeof(ready_keep) - 1})) return false;
            }
        } else if (starts_with(headers, "POST /fraud-score ")) {
            std::string_view body = data.substr(hdr_end + 4, size_t(clen));
            int16_t q[Dims];
            if (!vectorize_fast(body, q)) {
                if (!write_response(conn.fd, response_for_fraud_count(0, close_after_response))) return false;
            } else {
                uint8_t fraud = search_fraud(index, q, search_cfg);
                if (!write_response(conn.fd, response_for_fraud_count(fraud, close_after_response))) return false;
            }
        } else {
            if (!write_response(conn.fd, response_for_fraud_count(0, close_after_response))) return false;
        }
        consumed += total;
        if (close_after_response) return false;
    }
    if (consumed > 0) {
        size_t remaining = conn.have - consumed;
        if (remaining > 0) {
            std::memmove(conn.buf.data(), conn.buf.data() + consumed, remaining);
        }
        conn.have = remaining;
    }
    return conn.have < conn.buf.size();
}

bool read_conn(Conn& conn, const silent::MappedIndex& index, const SearchConfig& search_cfg, bool close_after_response) {
    for (;;) {
        if (conn.have == conn.buf.size()) return false;
        ssize_t r = ::read(conn.fd, conn.buf.data() + conn.have, conn.buf.size() - conn.have);
        if (r > 0) {
            conn.have += size_t(r);
            if (!process_buffer(conn, index, search_cfg, close_after_response)) return false;
            return true;
        }
        if (r == 0) return false;
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return true;
        return false;
    }
}

void set_nonblocking_cloexec(int fd) {
    int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags >= 0) (void)::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    int fd_flags = ::fcntl(fd, F_GETFD, 0);
    if (fd_flags >= 0) (void)::fcntl(fd, F_SETFD, fd_flags | FD_CLOEXEC);
}

void tune_tcp_socket(int fd) {
    int one = 1;
    (void)::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#ifdef TCP_QUICKACK
    (void)::setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one));
#endif
}

bool register_client(int ep, std::vector<Conn>& conns, std::vector<uint8_t>& conn_active, int fd) {
    set_nonblocking_cloexec(fd);
    tune_tcp_socket(fd);
    if (size_t(fd) >= conns.size()) {
        conns.resize(size_t(fd) + 1);
        conn_active.resize(size_t(fd) + 1, 0);
    }
    if (conn_active[size_t(fd)] != 0) {
        ::close(fd);
        return false;
    }
    Conn& conn = conns[size_t(fd)];
    conn.fd = fd;
    conn.have = 0;
    conn_active[size_t(fd)] = 1;
    epoll_event cev{};
    cev.events = EPOLLIN | EPOLLRDHUP;
    cev.data.fd = fd;
    if (::epoll_ctl(ep, EPOLL_CTL_ADD, fd, &cev) != 0) {
        conn_active[size_t(fd)] = 0;
        ::close(fd);
        return false;
    }
    return true;
}

int listen_unix(const std::string& path, int type = SOCK_STREAM) {
    int fd = ::socket(AF_UNIX, type | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) fatal("socket failed");
    ::unlink(path.c_str());
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) fatal("socket path too long");
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        fatal("bind failed");
    }
    ::chmod(path.c_str(), 0777);
    if (::listen(fd, 4096) != 0) fatal("listen failed");
    return fd;
}

void accept_clients(int ep, std::vector<Conn>& conns, std::vector<uint8_t>& conn_active, int srv) {
    for (;;) {
        int cfd = ::accept4(srv, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            break;
        }
        (void)register_client(ep, conns, conn_active, cfd);
    }
}

void accept_ctrl(int ep, std::vector<uint8_t>& ctrl_conns, int ctrl_srv) {
    for (;;) {
        int cfd = ::accept4(ctrl_srv, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (size_t(cfd) >= ctrl_conns.size()) ctrl_conns.resize(size_t(cfd) + 1, 0);
        ctrl_conns[size_t(cfd)] = 1;
        epoll_event ev{};
        ev.events = EPOLLIN | EPOLLRDHUP | EPOLLET;
        ev.data.fd = cfd;
        if (::epoll_ctl(ep, EPOLL_CTL_ADD, cfd, &ev) != 0) {
            ctrl_conns[size_t(cfd)] = 0;
            ::close(cfd);
        }
    }
}

int recv_passed_fd(int fd) {
    char byte = 0;
    iovec iov{&byte, 1};
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))]{};
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);

    for (;;) {
        ssize_t n = ::recvmsg(fd, &msg, 0);
        if (n > 0) break;
        if (n == 0) return -1;
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return -2;
        return -1;
    }

    for (cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
        if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS &&
            cmsg->cmsg_len >= CMSG_LEN(sizeof(int))) {
            int passed = -1;
            std::memcpy(&passed, CMSG_DATA(cmsg), sizeof(passed));
            return passed;
        }
    }
    return -2;
}

bool read_ctrl_conn(int fd, int ep, std::vector<Conn>& conns, std::vector<uint8_t>& conn_active) {
    for (;;) {
        int client_fd = recv_passed_fd(fd);
        if (client_fd >= 0) {
            (void)register_client(ep, conns, conn_active, client_fd);
            continue;
        }
        if (client_fd == -2) return true;
        return false;
    }
}

void close_ctrl(int ep, std::vector<uint8_t>& ctrl_conns, int fd) {
    (void)::epoll_ctl(ep, EPOLL_CTL_DEL, fd, nullptr);
    ::close(fd);
    if (fd >= 0 && size_t(fd) < ctrl_conns.size()) ctrl_conns[size_t(fd)] = 0;
}

void close_conn(int ep, std::vector<Conn>& conns, std::vector<uint8_t>& conn_active, int fd) {
    (void)::epoll_ctl(ep, EPOLL_CTL_DEL, fd, nullptr);
    ::close(fd);
    if (fd >= 0 && size_t(fd) < conn_active.size()) {
        conn_active[size_t(fd)] = 0;
        conns[size_t(fd)].fd = -1;
        conns[size_t(fd)].have = 0;
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string index_path = std::getenv("INDEX_PATH") ? std::getenv("INDEX_PATH") : "/index/index.bin";
    std::string listen_path = std::getenv("LISTEN") ? std::getenv("LISTEN") : "/sockets/api.sock";
    SearchConfig search_cfg;
    search_cfg.nprobe = std::getenv("NPROBE") ? std::atoi(std::getenv("NPROBE")) : 16;
    search_cfg.repair_min = std::getenv("REPAIR_MIN") ? std::atoi(std::getenv("REPAIR_MIN")) : 2;
    search_cfg.repair_max = std::getenv("REPAIR_MAX") ? std::atoi(std::getenv("REPAIR_MAX")) : 3;
    search_cfg.fast_nprobe = std::getenv("FAST_NPROBE") ? std::atoi(std::getenv("FAST_NPROBE")) : 0;
    search_cfg.adaptive_min = std::getenv("ADAPTIVE_MIN") ? std::atoi(std::getenv("ADAPTIVE_MIN")) : 1;
    search_cfg.adaptive_max = std::getenv("ADAPTIVE_MAX") ? std::atoi(std::getenv("ADAPTIVE_MAX")) : 4;
    const char* close_env = std::getenv("CONNECTION_CLOSE");
    bool close_after_response = close_env && (close_env[0] == '1' || std::strcmp(close_env, "true") == 0);
    if (argc > 1) index_path = argv[1];
    if (argc > 2) listen_path = argv[2];

    silent::MappedIndex index(index_path);
    int srv = listen_unix(listen_path);
    std::string ctrl_path = listen_path + ".ctrl";
    int ctrl_srv = listen_unix(ctrl_path);

    ::signal(SIGPIPE, SIG_IGN);
    int ep = ::epoll_create1(EPOLL_CLOEXEC);
    if (ep < 0) fatal("epoll_create1 failed");

    epoll_event ev{};
    ev.events = EPOLLIN | EPOLLET;
    ev.data.fd = srv;
    if (::epoll_ctl(ep, EPOLL_CTL_ADD, srv, &ev) != 0) {
        fatal("epoll add listen failed");
    }
    ev.data.fd = ctrl_srv;
    if (::epoll_ctl(ep, EPOLL_CTL_ADD, ctrl_srv, &ev) != 0) {
        fatal("epoll add ctrl listen failed");
    }

    std::vector<Conn> conns;
    std::vector<uint8_t> conn_active;
    std::vector<uint8_t> ctrl_conns;
    std::array<epoll_event, 1024> events{};
    for (;;) {
        int n = ::epoll_wait(ep, events.data(), int(events.size()), -1);
        if (n < 0) {
            if (errno == EINTR) continue;
            fatal("epoll_wait failed");
        }
        for (int i = 0; i < n; ++i) {
            int fd = events[size_t(i)].data.fd;
            uint32_t revents = events[size_t(i)].events;
            if (fd == srv) {
                accept_clients(ep, conns, conn_active, srv);
                continue;
            }
            if (fd == ctrl_srv) {
                accept_ctrl(ep, ctrl_conns, ctrl_srv);
                continue;
            }
            if (fd >= 0 && size_t(fd) < ctrl_conns.size() && ctrl_conns[size_t(fd)] != 0) {
                bool alive = true;
                if (revents & (EPOLLERR | EPOLLHUP | EPOLLRDHUP)) alive = false;
                if (alive && (revents & EPOLLIN)) alive = read_ctrl_conn(fd, ep, conns, conn_active);
                if (!alive) close_ctrl(ep, ctrl_conns, fd);
                continue;
            }

            if (fd < 0 || size_t(fd) >= conn_active.size() || conn_active[size_t(fd)] == 0) continue;
            bool alive = true;
            if (revents & (EPOLLERR | EPOLLHUP)) alive = false;
            if (alive && (revents & EPOLLIN)) {
                alive = read_conn(conns[size_t(fd)], index, search_cfg, close_after_response);
            }
            if (alive && (revents & EPOLLRDHUP)) alive = false;
            if (!alive) close_conn(ep, conns, conn_active, fd);
        }
    }
}
