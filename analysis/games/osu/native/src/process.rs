//! Process discovery.
//!
//! Linux/wine: osu! shows up with ``comm == "osu!.exe"`` (wine forwards
//! the Windows exe name into the kernel's comm field); we walk /proc.
//!
//! Windows: native osu!.exe. We walk the Toolhelp32 process snapshot
//! and match ``szExeFile``. Same semantics as tosu's ``find_processes``
//! in ``packages/tsprocess/lib/memory/memory_windows.cc``.
//!
//! Either call runs at most once per connect (not per tick), so the
//! per-call cost of the walk is a non-issue.

#[cfg(target_os = "linux")]
pub fn find_osu_pid_linux() -> Option<u32> {
    use std::fs;
    let entries = fs::read_dir("/proc").ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // /proc has non-numeric entries too (self, cpuinfo, …) — skip.
        let Ok(pid) = name.parse::<u32>() else {
            continue;
        };
        // ``comm`` is the kernel's short process name (TASK_COMM_LEN,
        // typically 16 chars). For wine-launched osu!, it's exactly
        // "osu!.exe" — we match strictly so random processes with
        // "osu" in them don't false-positive.
        let comm_path = format!("/proc/{pid}/comm");
        let Ok(mut comm) = fs::read_to_string(&comm_path) else {
            continue;
        };
        if comm.ends_with('\n') {
            comm.pop();
        }
        if comm == "osu!.exe" || comm == "osu!" {
            return Some(pid);
        }
    }
    None
}

#[cfg(target_os = "windows")]
pub fn find_osu_pid_windows() -> Option<u32> {
    use windows::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W,
        TH32CS_SNAPPROCESS,
    };

    // SAFETY: CreateToolhelp32Snapshot returns an INVALID_HANDLE_VALUE
    // on failure (wrapped as an Err by the `windows` crate). On success
    // we own a HANDLE that is closed via Drop.
    let snap = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) }.ok()?;

    let mut entry = PROCESSENTRY32W {
        dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32,
        ..Default::default()
    };

    // SAFETY: `entry` is initialized with the correct `dwSize`; the API
    // writes into the rest of the struct on success. `snap` is valid
    // (checked above).
    if unsafe { Process32FirstW(snap, &mut entry) }.is_err() {
        return None;
    }

    loop {
        // szExeFile is a fixed-size [u16; 260] null-terminated UTF-16
        // array. Truncate at the null before decoding so we don't lift
        // trailing garbage bytes into the string.
        let nul = entry
            .szExeFile
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(entry.szExeFile.len());
        let name = String::from_utf16_lossy(&entry.szExeFile[..nul]);
        if name.eq_ignore_ascii_case("osu!.exe") {
            return Some(entry.th32ProcessID);
        }
        // SAFETY: same as Process32FirstW above.
        if unsafe { Process32NextW(snap, &mut entry) }.is_err() {
            return None;
        }
    }
}
