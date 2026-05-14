#define _GNU_SOURCE

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define MAX_UPSTREAMS 8
#define BACKLOG 65535

typedef struct {
    char path[108];
    int fd;

    char byte;
    struct iovec iov;
    union {
        struct cmsghdr align;
        char buf[CMSG_SPACE(sizeof(int))];
    } control;
    struct msghdr msg;
    struct cmsghdr* cmsg;
} upstream_t;

static upstream_t upstreams[MAX_UPSTREAMS];
static int upstream_count = 0;
static uint32_t rr_next = 0;

static void die(const char* msg) {
    perror(msg);
    _Exit(1);
}

static void sleep_ms(long ms) {
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000L;
    while (nanosleep(&ts, &ts) < 0 && errno == EINTR) {}
}

static int set_nonblock_cloexec(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0) (void)fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    flags = fcntl(fd, F_GETFD, 0);
    if (flags >= 0) (void)fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
    return 0;
}

static void init_msg(upstream_t* u) {
    memset(&u->control, 0, sizeof(u->control));
    memset(&u->msg, 0, sizeof(u->msg));

    u->byte = 1;
    u->iov.iov_base = &u->byte;
    u->iov.iov_len = 1;

    u->msg.msg_iov = &u->iov;
    u->msg.msg_iovlen = 1;
    u->msg.msg_control = u->control.buf;
    u->msg.msg_controllen = sizeof(u->control.buf);

    u->cmsg = CMSG_FIRSTHDR(&u->msg);
    if (u->cmsg == NULL) _Exit(1);
    u->cmsg->cmsg_level = SOL_SOCKET;
    u->cmsg->cmsg_type = SCM_RIGHTS;
    u->cmsg->cmsg_len = CMSG_LEN(sizeof(int));
}

static void add_upstream(const char* begin, size_t len) {
    while (len > 0 && (*begin == ' ' || *begin == '\t')) {
        ++begin;
        --len;
    }
    while (len > 0 && (begin[len - 1] == ' ' || begin[len - 1] == '\t' || begin[len - 1] == '\n')) {
        --len;
    }
    if (len == 0 || upstream_count >= MAX_UPSTREAMS) return;

    upstream_t* u = &upstreams[upstream_count++];
    memset(u, 0, sizeof(*u));
    if (len + 5 >= sizeof(u->path)) _Exit(1);
    memcpy(u->path, begin, len);
    if (len >= 5 && memcmp(u->path + len - 5, ".ctrl", 5) == 0) {
        u->path[len] = '\0';
    } else {
        memcpy(u->path + len, ".ctrl", 6);
    }
    u->fd = -1;
    init_msg(u);
}

static void parse_upstreams(void) {
    const char* env = getenv("UPSTREAMS");
    if (env == NULL || env[0] == '\0') env = "/sockets/api1.sock,/sockets/api2.sock";
    const char* p = env;
    const char* start = p;
    for (;; ++p) {
        if (*p == ',' || *p == '\0') {
            add_upstream(start, (size_t)(p - start));
            if (*p == '\0') break;
            start = p + 1;
        }
    }
    if (upstream_count == 0) _Exit(1);
}

static int connect_ctrl_once(const char* path) {
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;

    int sndbuf = 256 * 1024;
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    set_nonblock_cloexec(fd);
    return fd;
}

static int connect_ctrl_wait(const char* path) {
    for (;;) {
        int fd = connect_ctrl_once(path);
        if (fd >= 0) return fd;
        sleep_ms(10);
    }
}

static void connect_all(void) {
    for (int i = 0; i < upstream_count; ++i) {
        upstreams[i].fd = connect_ctrl_wait(upstreams[i].path);
    }
}

static int reconnect_one(int idx) {
    if (upstreams[idx].fd >= 0) close(upstreams[idx].fd);
    upstreams[idx].fd = -1;
    for (int tries = 0; tries < 10; ++tries) {
        int fd = connect_ctrl_once(upstreams[idx].path);
        if (fd >= 0) {
            upstreams[idx].fd = fd;
            return 0;
        }
        sleep_ms(2);
    }
    return -1;
}

static int is_broken_ctrl_errno(int e) {
    return e == EPIPE || e == ECONNRESET || e == ENOTCONN || e == EBADF;
}

static int send_fd_once(upstream_t* u, int client_fd) {
    u->msg.msg_controllen = sizeof(u->control.buf);
    memcpy(CMSG_DATA(u->cmsg), &client_fd, sizeof(client_fd));

    for (;;) {
        ssize_t n = sendmsg(u->fd, &u->msg, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (n == 1) return 0;
        if (n < 0 && errno == EINTR) continue;
        return errno ? errno : EIO;
    }
}

static int handoff(int idx, int client_fd) {
    if (upstreams[idx].fd < 0 && reconnect_one(idx) != 0) return ECONNREFUSED;
    int e = send_fd_once(&upstreams[idx], client_fd);
    if (e == 0) return 0;
    if (is_broken_ctrl_errno(e)) {
        if (reconnect_one(idx) != 0) return e;
        e = send_fd_once(&upstreams[idx], client_fd);
    }
    return e;
}

static int handoff_any(int first, int client_fd) {
    if (upstream_count == 2) {
        int e = handoff(first, client_fd);
        if (e == 0) return 0;
        return handoff(first ^ 1, client_fd);
    }

    for (int offset = 0; offset < upstream_count; ++offset) {
        int idx = first + offset;
        if (idx >= upstream_count) idx -= upstream_count;
        int e = handoff(idx, client_fd);
        if (e == 0) return 0;
    }

    return -1;
}

static int listen_tcp(int port) {
    int fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;

    int one = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &one, sizeof(one));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t)port);

    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    if (listen(fd, BACKLOG) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);
    parse_upstreams();
    connect_all();

    int port = 9999;
    const char* port_env = getenv("PORT");
    if (port_env != NULL && port_env[0] != '\0') port = atoi(port_env);

    int server_fd = listen_tcp(port);
    if (server_fd < 0) die("listen");

    for (;;) {
        int client_fd = accept4(server_fd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                struct pollfd pfd;
                pfd.fd = server_fd;
                pfd.events = POLLIN;
                pfd.revents = 0;
                (void)poll(&pfd, 1, -1);
                continue;
            }
            continue;
        }

        uint32_t rr = rr_next++;
        int first = upstream_count == 2 ? (int)(rr & 1u) : (int)(rr % (uint32_t)upstream_count);
        (void)handoff_any(first, client_fd);
        close(client_fd);
    }
}
