// Log lines go to %TEMP%\vsrg-gl.log (or $VSRG_GL_LOG if set) and stderr.
// Tail the file from any shell:  Get-Content -Wait $env:TEMP\vsrg-gl.log

#pragma once

#ifdef __cplusplus
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <ctime>
extern "C" {
#else
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#endif

static inline const char *vsrg_log_path(void) {
    const char *override = getenv("VSRG_GL_LOG");
    if (override && *override) return override;
    static char path[512];
    if (path[0]) return path;
    const char *tmp = getenv("TEMP");
    if (!tmp || !*tmp) tmp = getenv("TMP");
    if (!tmp || !*tmp) tmp = ".";
    snprintf(path, sizeof(path), "%s\\vsrg-gl.log", tmp);
    return path;
}

// Opened/appended/closed per line so a crash still leaves a full trace.
// Only called on state transitions (hook install, shm attach, widget-
// count change), not every frame, so fopen cost is a non-issue.
static inline void vsrg_log_raw(const char *fmt, ...) {
    char line[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(line, sizeof(line) - 1, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n >= sizeof(line) - 1) n = (int)(sizeof(line) - 2);
    if (n == 0 || line[n - 1] != '\n') {
        line[n++] = '\n';
        line[n] = '\0';
    }

    char stamp[32];
    time_t t = time(NULL);
    struct tm lt;
#ifdef _WIN32
    localtime_s(&lt, &t);
#else
    lt = *localtime(&t);
#endif
    int sn = (int)strftime(stamp, sizeof(stamp), "%H:%M:%S ", &lt);
    if (sn <= 0) stamp[0] = '\0';

    FILE *f = fopen(vsrg_log_path(), "a");
    if (f) {
        fputs(stamp, f);
        fputs(line, f);
        fclose(f);
    }
    fputs(stamp, stderr);
    fputs(line, stderr);
}

#define VSRG_LOG(fmt, ...) vsrg_log_raw(fmt "\n", ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif
