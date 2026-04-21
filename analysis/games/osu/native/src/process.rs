//! Process discovery.
//!
//! On Linux/wine, osu! shows up with ``comm == "osu!.exe"`` (wine
//! forwards the Windows exe name into the kernel's comm field). We
//! walk /proc once per call; this runs at most once per connect, not
//! per tick, so the cost is fine.

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
