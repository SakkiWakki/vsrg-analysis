// Web-texture host: bring-up + per-frame maintenance.
//
// Architecture:
//
//   [listener thread]                       [render thread (swap hook)]
//     accept()                                web_texture_host_tick()
//     recvmsg w/ SCM_RIGHTS                     drain pending queue
//     decode VsrgWebTexFrame                    for each fd:
//     lock q; push (frame + fd); unlock           eglCreateImage(dmabuf attrs)
//                                                 gl create tex + bind
//                                                 glEGLImageTargetTexture2DOES
//                                                 store in cache[chan][gen]
//                                                 close fd
//
//   [render thread, inside vsrg_draw_widgets]
//     resolver(chan, gen) -> lookup cache
//
// GL calls MUST happen on the render thread because EGL is
// context-current-bound. The listener thread is intentionally
// GL-ignorant -- it only knows how to recvmsg.
//
// Cache shape:
//
//   For each channel we hold at most ONE imported texture -- the
//   newest generation. When gen bumps, we delete the prior one and
//   rebind. This matches the producer's "latest frame wins" model
//   and keeps total GPU memory bounded at O(N channels), not O(N
//   channels × history).
//
// Concurrency:
//
//   - pending_q: protected by a small pthread mutex. Lock is held
//     only long enough to push/pop a struct; never while doing GL
//     work or sendmsg.
//   - cache: touched exclusively from the render thread. No lock
//     needed; the resolver runs on the same thread as web_texture_host_tick.
//   - resolver installation: __atomic_store with release, picked up
//     by widgets.c via __atomic_load acquire.

#define _GNU_SOURCE
#include "web_texture_host.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <GL/glext.h>

#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include "overlay_shm.h"
#include "web_texture_ipc.h"
#include "widgets.h"

// ── Cache entry ────────────────────────────────────────────────────

#define MAX_CHANNELS 16

typedef struct {
    int      in_use;
    uint32_t channel_id;
    uint32_t generation;
    GLuint   gl_tex;
    int      width;
    int      height;
    EGLImageKHR egl_image;     // kept so we can destroy it on replace
} CacheEntry;

static CacheEntry g_cache[MAX_CHANNELS];

// ── Pending queue ──────────────────────────────────────────────────

typedef struct {
    VsrgWebTexFrame frame;
    int fd;             // -1 for RELEASE (no fd)
} PendingMsg;

#define QUEUE_CAP 32

static pthread_mutex_t g_q_mu = PTHREAD_MUTEX_INITIALIZER;
static PendingMsg      g_q[QUEUE_CAP];
static int             g_q_head = 0;     // next read slot
static int             g_q_tail = 0;     // next write slot

// ── Listener thread state ─────────────────────────────────────────

static pthread_t      g_listener;
static atomic_int     g_listener_started = 0;
static atomic_int     g_stop             = 0;
static int            g_srv_fd           = -1;   // listening socket
static int            g_conn_fd          = -1;   // accepted socket (one at a time)

// ── EGL function pointers (resolved once on the render thread) ────

static PFNEGLCREATEIMAGEKHRPROC      p_eglCreateImageKHR      = NULL;
static PFNEGLDESTROYIMAGEKHRPROC     p_eglDestroyImageKHR     = NULL;
static PFNGLEGLIMAGETARGETTEXTURE2DOESPROC p_glEGLImageTargetTexture2DOES = NULL;
static int                           g_egl_bound              = 0;

static int bind_egl_symbols(void) {
    if (g_egl_bound) return 1;
    p_eglCreateImageKHR =
        (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
    p_eglDestroyImageKHR =
        (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
    p_glEGLImageTargetTexture2DOES =
        (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)
            eglGetProcAddress("glEGLImageTargetTexture2DOES");
    if (!p_eglCreateImageKHR
        || !p_eglDestroyImageKHR
        || !p_glEGLImageTargetTexture2DOES) {
        fprintf(stderr,
                "[webtex] missing EGL/GL extensions; "
                "dmabuf import unavailable\n");
        return 0;
    }
    g_egl_bound = 1;
    return 1;
}

// ── Cache helpers (render thread only) ─────────────────────────────

static CacheEntry *cache_find(uint32_t channel_id) {
    for (int i = 0; i < MAX_CHANNELS; i++) {
        if (g_cache[i].in_use && g_cache[i].channel_id == channel_id) {
            return &g_cache[i];
        }
    }
    return NULL;
}

static CacheEntry *cache_alloc(uint32_t channel_id) {
    for (int i = 0; i < MAX_CHANNELS; i++) {
        if (!g_cache[i].in_use) {
            g_cache[i].in_use = 1;
            g_cache[i].channel_id = channel_id;
            g_cache[i].generation = 0;
            g_cache[i].gl_tex = 0;
            g_cache[i].egl_image = EGL_NO_IMAGE_KHR;
            return &g_cache[i];
        }
    }
    return NULL;
}

static void cache_release(CacheEntry *e) {
    if (!e || !e->in_use) return;
    EGLDisplay dpy = eglGetCurrentDisplay();
    if (e->egl_image != EGL_NO_IMAGE_KHR && dpy != EGL_NO_DISPLAY
        && p_eglDestroyImageKHR) {
        p_eglDestroyImageKHR(dpy, e->egl_image);
    }
    if (e->gl_tex != 0) {
        glDeleteTextures(1, &e->gl_tex);
    }
    memset(e, 0, sizeof(*e));
}

// ── Resolver callback (installed on widgets.c) ────────────────────

static int resolver(uint32_t channel_id, uint32_t generation,
                    uint32_t *out_gl_tex, int *out_tex_w, int *out_tex_h) {
    CacheEntry *e = cache_find(channel_id);
    if (!e || e->generation != generation) {
        // Stale: the widget references a generation we haven't
        // imported (yet, or we're behind). Draw nothing.
        return 0;
    }
    *out_gl_tex = e->gl_tex;
    *out_tex_w  = e->width;
    *out_tex_h  = e->height;
    return 1;
}

// ── dmabuf import (render thread) ─────────────────────────────────

static int import_dmabuf(const VsrgWebTexFrame *f, int fd, CacheEntry *e) {
    EGLDisplay dpy = eglGetCurrentDisplay();
    if (dpy == EGL_NO_DISPLAY) {
        fprintf(stderr, "[webtex] no current EGL display on import\n");
        return 0;
    }

    // Build the EGL_LINUX_DMA_BUF_EXT attribute list. Single-plane
    // RGB; NV12/etc. would need per-plane (fd, offset, pitch, mod lo,
    // mod hi) triplets for planes 1..n.
    EGLint attrs[] = {
        EGL_WIDTH,                     (EGLint)f->width,
        EGL_HEIGHT,                    (EGLint)f->height,
        EGL_LINUX_DRM_FOURCC_EXT,      (EGLint)f->format,
        EGL_DMA_BUF_PLANE0_FD_EXT,     fd,
        EGL_DMA_BUF_PLANE0_OFFSET_EXT, (EGLint)f->offsets[0],
        EGL_DMA_BUF_PLANE0_PITCH_EXT,  (EGLint)f->strides[0],
        EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
            (EGLint)(f->modifier & 0xFFFFFFFFu),
        EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
            (EGLint)((f->modifier >> 32) & 0xFFFFFFFFu),
        EGL_NONE,
    };

    EGLImageKHR img = p_eglCreateImageKHR(
        dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT,
        /*buffer=*/(EGLClientBuffer)NULL, attrs);
    if (img == EGL_NO_IMAGE_KHR) {
        fprintf(stderr, "[webtex] eglCreateImage failed (egl=0x%x) "
                        "w=%u h=%u fmt=0x%08x mod=0x%llx\n",
                eglGetError(),
                f->width, f->height, f->format,
                (unsigned long long)f->modifier);
        return 0;
    }

    // Rebind into a fresh GL_TEXTURE_2D. Re-use the cache slot's
    // existing GL name if possible; otherwise allocate one.
    GLuint tex = e->gl_tex;
    if (tex == 0) {
        glGenTextures(1, &tex);
    }
    glBindTexture(GL_TEXTURE_2D, tex);
    p_glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, (GLeglImageOES)img);
    // Linear sampling; no mipmaps (source texture is single-level).
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    GLenum gerr = glGetError();
    if (gerr != GL_NO_ERROR) {
        fprintf(stderr, "[webtex] glEGLImageTargetTexture2DOES failed "
                        "gl_err=0x%x\n", gerr);
        p_eglDestroyImageKHR(dpy, img);
        if (e->gl_tex == 0) glDeleteTextures(1, &tex);
        return 0;
    }

    // If the slot already held a prior generation, free its image.
    if (e->egl_image != EGL_NO_IMAGE_KHR) {
        p_eglDestroyImageKHR(dpy, e->egl_image);
    }
    e->gl_tex     = tex;
    e->egl_image  = img;
    e->generation = f->generation;
    e->width      = (int)f->width;
    e->height     = (int)f->height;
    return 1;
}

// ── Listener thread ───────────────────────────────────────────────

static ssize_t recv_with_fd(int sock, VsrgWebTexFrame *out_frame,
                            int *out_fd) {
    char cmsg_buf[CMSG_SPACE(sizeof(int))];
    struct iovec iov = {
        .iov_base = out_frame,
        .iov_len  = sizeof(*out_frame),
    };
    struct msghdr msg = {
        .msg_iov        = &iov,
        .msg_iovlen     = 1,
        .msg_control    = cmsg_buf,
        .msg_controllen = sizeof(cmsg_buf),
    };
    ssize_t n = recvmsg(sock, &msg, 0);
    if (n <= 0) return n;

    *out_fd = -1;
    for (struct cmsghdr *c = CMSG_FIRSTHDR(&msg); c != NULL;
         c = CMSG_NXTHDR(&msg, c)) {
        if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS
            && c->cmsg_len >= CMSG_LEN(sizeof(int))) {
            memcpy(out_fd, CMSG_DATA(c), sizeof(int));
            break;
        }
    }
    return n;
}

static void queue_push(const VsrgWebTexFrame *f, int fd) {
    pthread_mutex_lock(&g_q_mu);
    int next_tail = (g_q_tail + 1) % QUEUE_CAP;
    if (next_tail == g_q_head) {
        // Full. Drop the oldest to make room (and close its fd) so
        // the freshest frames always reach the import pass.
        if (g_q[g_q_head].fd >= 0) close(g_q[g_q_head].fd);
        g_q_head = (g_q_head + 1) % QUEUE_CAP;
    }
    g_q[g_q_tail].frame = *f;
    g_q[g_q_tail].fd    = fd;
    g_q_tail = next_tail;
    pthread_mutex_unlock(&g_q_mu);
}

static int queue_pop(PendingMsg *out) {
    pthread_mutex_lock(&g_q_mu);
    if (g_q_head == g_q_tail) {
        pthread_mutex_unlock(&g_q_mu);
        return 0;
    }
    *out = g_q[g_q_head];
    g_q_head = (g_q_head + 1) % QUEUE_CAP;
    pthread_mutex_unlock(&g_q_mu);
    return 1;
}

static void *listener_main(void *arg) {
    (void)arg;
    while (!atomic_load_explicit(&g_stop, memory_order_acquire)) {
        int cfd = accept(g_srv_fd, NULL, NULL);
        if (cfd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK
                || errno == EINTR) continue;
            break;
        }
        g_conn_fd = cfd;
        while (!atomic_load_explicit(&g_stop, memory_order_acquire)) {
            VsrgWebTexFrame f;
            int fd = -1;
            ssize_t n = recv_with_fd(cfd, &f, &fd);
            if (n <= 0) break;
            if ((size_t)n < sizeof(f)) {
                if (fd >= 0) close(fd);
                continue;
            }
            if (f.magic != VSRG_WEB_TEX_MAGIC
                || f.version != VSRG_WEB_TEX_VERSION) {
                if (fd >= 0) close(fd);
                continue;
            }
            queue_push(&f, fd);
        }
        close(cfd);
        g_conn_fd = -1;
    }
    return NULL;
}

// ── Public API ────────────────────────────────────────────────────

int web_texture_host_start(void) {
    if (atomic_exchange(&g_listener_started, 1)) return 1;

    // Unlink any stale socket left by a previous run. If bind fails
    // because another overlay is already up, we fall through without
    // crashing -- web textures just stay off.
    unlink(VSRG_WEB_TEX_SOCKET_PATH);

    g_srv_fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (g_srv_fd < 0) {
        perror("[webtex] socket");
        atomic_store(&g_listener_started, 0);
        return 0;
    }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, VSRG_WEB_TEX_SOCKET_PATH,
            sizeof(addr.sun_path) - 1);
    if (bind(g_srv_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("[webtex] bind");
        close(g_srv_fd); g_srv_fd = -1;
        atomic_store(&g_listener_started, 0);
        return 0;
    }
    chmod(VSRG_WEB_TEX_SOCKET_PATH, 0600);
    if (listen(g_srv_fd, 1) < 0) {
        perror("[webtex] listen");
        close(g_srv_fd); g_srv_fd = -1;
        atomic_store(&g_listener_started, 0);
        return 0;
    }

    if (pthread_create(&g_listener, NULL, listener_main, NULL) != 0) {
        perror("[webtex] pthread_create");
        close(g_srv_fd); g_srv_fd = -1;
        atomic_store(&g_listener_started, 0);
        return 0;
    }

    vsrg_set_web_texture_resolver(resolver);
    fprintf(stderr, "[webtex] listening at %s\n",
            VSRG_WEB_TEX_SOCKET_PATH);
    return 1;
}

void web_texture_host_tick(void) {
    if (!atomic_load(&g_listener_started)) return;
    if (!bind_egl_symbols()) return;

    PendingMsg m;
    while (queue_pop(&m)) {
        if (m.frame.kind == VSRG_WEB_TEX_KIND_RELEASE) {
            CacheEntry *e = cache_find(m.frame.channel_id);
            if (e) cache_release(e);
            // RELEASE shouldn't carry an fd but be defensive.
            if (m.fd >= 0) close(m.fd);
            continue;
        }
        if (m.frame.kind != VSRG_WEB_TEX_KIND_PUBLISH) {
            if (m.fd >= 0) close(m.fd);
            continue;
        }
        if (m.fd < 0) {
            // PUBLISH without fd is a protocol error; skip.
            continue;
        }

        CacheEntry *e = cache_find(m.frame.channel_id);
        if (!e) e = cache_alloc(m.frame.channel_id);
        if (!e) {
            // Out of cache slots -- drop.
            close(m.fd);
            continue;
        }
        // import_dmabuf takes ownership of the fd's lifetime inside
        // EGL; we always close our reference regardless of whether
        // the import succeeded.
        (void)import_dmabuf(&m.frame, m.fd, e);
        close(m.fd);
    }
}

void web_texture_host_shutdown(void) {
    if (!atomic_exchange(&g_stop, 1)) {
        // Wake the listener's blocking accept() by closing its fd
        // out from under it; on Linux that makes accept return EBADF.
        if (g_srv_fd >= 0) {
            shutdown(g_srv_fd, SHUT_RDWR);
            close(g_srv_fd);
            g_srv_fd = -1;
        }
        if (g_conn_fd >= 0) {
            shutdown(g_conn_fd, SHUT_RDWR);
        }
    }
    if (atomic_load(&g_listener_started)) {
        pthread_join(g_listener, NULL);
    }
    vsrg_set_web_texture_resolver(NULL);

    // Release cached textures. We can only free GL resources with a
    // live context; if called from atexit after the GL context is
    // gone, leak the textures (the process is exiting anyway).
    for (int i = 0; i < MAX_CHANNELS; i++) {
        cache_release(&g_cache[i]);
    }

    // Drain any unconsumed queued fds.
    PendingMsg m;
    while (queue_pop(&m)) {
        if (m.fd >= 0) close(m.fd);
    }

    unlink(VSRG_WEB_TEX_SOCKET_PATH);
    atomic_store(&g_listener_started, 0);
    atomic_store(&g_stop, 0);
}
