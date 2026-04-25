// Minimal DLL injector. CreateRemoteThread + LoadLibraryW into a target
// PID or exe name. VSRG_GL_OVERLAY must be set on the TARGET's environment
// (we don't spawn it here, we attach to a running process), so launch osu!
// from a shell where that var is already set.
//
// Usage: inject <pid|exe-name> <absolute-path-to-dll>

#include <windows.h>
#include <tlhelp32.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

static DWORD find_pid_by_name(const char *exe_name) {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) return 0;

    PROCESSENTRY32 pe = { .dwSize = sizeof(pe) };
    DWORD pid = 0;
    if (Process32First(snap, &pe)) {
        do {
            if (_stricmp(pe.szExeFile, exe_name) == 0) {
                pid = pe.th32ProcessID;
                break;
            }
        } while (Process32Next(snap, &pe));
    }
    CloseHandle(snap);
    return pid;
}

static int inject(DWORD pid, const wchar_t *dll_path_w) {
    HANDLE proc = OpenProcess(
        PROCESS_CREATE_THREAD | PROCESS_VM_OPERATION |
        PROCESS_VM_WRITE | PROCESS_VM_READ | PROCESS_QUERY_INFORMATION,
        FALSE, pid);
    if (!proc) {
        fprintf(stderr, "OpenProcess(pid=%lu) failed: %lu\n",
                pid, GetLastError());
        return 1;
    }

    // kernel32.dll is at the same address in every process on a given
    // boot, so LoadLibraryW's address in us matches its address in the
    // target. That's what makes CreateRemoteThread(LoadLibraryW) work.
    HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
    FARPROC load_lib = GetProcAddress(k32, "LoadLibraryW");
    if (!load_lib) {
        fprintf(stderr, "GetProcAddress(LoadLibraryW) failed: %lu\n",
                GetLastError());
        CloseHandle(proc);
        return 1;
    }

    size_t path_bytes = (wcslen(dll_path_w) + 1) * sizeof(wchar_t);
    LPVOID remote_buf = VirtualAllocEx(proc, NULL, path_bytes,
                                       MEM_COMMIT | MEM_RESERVE,
                                       PAGE_READWRITE);
    if (!remote_buf) {
        fprintf(stderr, "VirtualAllocEx failed: %lu\n", GetLastError());
        CloseHandle(proc);
        return 1;
    }

    if (!WriteProcessMemory(proc, remote_buf, dll_path_w, path_bytes, NULL)) {
        fprintf(stderr, "WriteProcessMemory failed: %lu\n", GetLastError());
        VirtualFreeEx(proc, remote_buf, 0, MEM_RELEASE);
        CloseHandle(proc);
        return 1;
    }

    HANDLE thr = CreateRemoteThread(proc, NULL, 0,
                                    (LPTHREAD_START_ROUTINE)load_lib,
                                    remote_buf, 0, NULL);
    if (!thr) {
        fprintf(stderr, "CreateRemoteThread failed: %lu\n", GetLastError());
        VirtualFreeEx(proc, remote_buf, 0, MEM_RELEASE);
        CloseHandle(proc);
        return 1;
    }

    WaitForSingleObject(thr, INFINITE);

    // Thread exit code = LoadLibraryW's return = HMODULE (or NULL on
    // failure). Truncation to 32-bit is fine for the null check.
    DWORD exit_code = 0;
    GetExitCodeThread(thr, &exit_code);

    CloseHandle(thr);
    VirtualFreeEx(proc, remote_buf, 0, MEM_RELEASE);
    CloseHandle(proc);

    if (exit_code == 0) {
        fprintf(stderr,
                "LoadLibraryW returned NULL in target ; DLL failed to load "
                "(check bitness match, missing deps, DllMain errors)\n");
        return 1;
    }

    printf("injected: HMODULE=0x%lx in pid %lu\n", exit_code, pid);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr,
                "usage: %s <pid|exe-name> <absolute-dll-path>\n"
                "  e.g. %s osu!.exe C:\\path\\to\\vsrg_gl_overlay.dll\n",
                argv[0], argv[0]);
        return 2;
    }

    const char *target = argv[1];
    const char *dll = argv[2];

    DWORD pid = (DWORD)strtoul(target, NULL, 10);
    if (pid == 0) {
        pid = find_pid_by_name(target);
        if (pid == 0) {
            fprintf(stderr, "no process named '%s'\n", target);
            return 1;
        }
        printf("resolved '%s' -> pid %lu\n", target, pid);
    }

    wchar_t dll_w[MAX_PATH];
    int n = MultiByteToWideChar(CP_UTF8, 0, dll, -1, dll_w, MAX_PATH);
    if (n == 0) {
        fprintf(stderr, "dll path too long or invalid utf-8\n");
        return 1;
    }

    // Relative paths resolve against the target's cwd, not ours ; reject
    // them up front rather than debug a silent NULL from the remote thread.
    if (dll_w[0] == 0 || (dll_w[1] != L':' && dll_w[0] != L'\\')) {
        fprintf(stderr, "dll path must be absolute: %s\n", dll);
        return 1;
    }

    return inject(pid, dll_w);
}
