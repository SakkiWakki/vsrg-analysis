//! Windows-specific memory read + pattern scanning. Mirrors
//! ``linux_mem``'s API: ``read_exact``, ``scan``, ``scan_all``.
//!
//! Same same-user, no-admin approach as tosu's
//! ``packages/tsprocess/lib/memory/memory_windows.cc``. WoW64 handles
//! 64-bit-reader / 32-bit-target transparently for reads ≤ 4 GiB.

use std::io::{self, ErrorKind};

use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::Diagnostics::Debug::ReadProcessMemory;
use windows::Win32::System::Memory::{
    VirtualQueryEx, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_IMAGE, PAGE_EXECUTE_READ,
    PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_READONLY, PAGE_READWRITE,
    PAGE_WRITECOPY,
};
use windows::Win32::System::Threading::{
    OpenProcess, PROCESS_QUERY_INFORMATION, PROCESS_VM_READ,
};

use crate::pattern::Pattern;

/// The `windows` crate's `HANDLE` is `Copy` and doesn't close on drop,
/// so we wrap it to guarantee `CloseHandle` on every return path.
struct Process(HANDLE);

impl Process {
    fn open(pid: u32) -> io::Result<Self> {
        // SAFETY: OpenProcess is FFI-safe; the `windows` crate turns a
        // NULL return into an Err so we never construct Self on failure.
        let h = unsafe {
            OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, false, pid)
        }
        .map_err(|e| io::Error::new(ErrorKind::PermissionDenied, e.to_string()))?;
        Ok(Self(h))
    }

    fn handle(&self) -> HANDLE {
        self.0
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        // SAFETY: `Self` is only constructed from a valid OpenProcess
        // return; Drop runs once.
        let _ = unsafe { CloseHandle(self.0) };
    }
}

/// Copy ``buf.len()`` bytes from ``pid``'s address space at ``addr``.
pub fn read_exact(pid: u32, addr: u64, buf: &mut [u8]) -> io::Result<()> {
    let proc = Process::open(pid)?;
    read_exact_with(proc.handle(), addr, buf)
}

fn read_exact_with(handle: HANDLE, addr: u64, buf: &mut [u8]) -> io::Result<()> {
    let mut out_read: usize = 0;
    // SAFETY:
    //   - buf.as_mut_ptr() is valid for buf.len() bytes (from &mut [u8]).
    //   - `out_read` is a live stack local.
    //   - The remote address is handed to the kernel, never deref'd here.
    let ok = unsafe {
        ReadProcessMemory(
            handle,
            addr as *const _,
            buf.as_mut_ptr() as *mut _,
            buf.len(),
            Some(&mut out_read),
        )
    };
    if ok.is_err() {
        return Err(io::Error::new(
            ErrorKind::Other,
            format!("ReadProcessMemory failed at 0x{addr:x}"),
        ));
    }
    if out_read != buf.len() {
        return Err(io::Error::new(
            ErrorKind::UnexpectedEof,
            format!("ReadProcessMemory short read: {out_read}/{}", buf.len()),
        ));
    }
    Ok(())
}

struct Region {
    start: u64,
    end: u64,
}

// Comparing `.0` bits directly avoids depending on BitAnd impls that
// come and go across `windows` crate versions.
fn region_kept(info: &MEMORY_BASIC_INFORMATION) -> bool {
    if info.State.0 & MEM_COMMIT.0 == 0 {
        return false;
    }
    // Superset of tosu's filter: also keep executable PE image regions
    // (MEM_IMAGE + PAGE_EXECUTE_READ) where static osu! code lives.
    const READABLE: u32 = PAGE_READWRITE.0
        | PAGE_WRITECOPY.0
        | PAGE_EXECUTE_READWRITE.0
        | PAGE_EXECUTE_WRITECOPY.0
        | PAGE_EXECUTE_READ.0
        | PAGE_READONLY.0;
    if info.Protect.0 & READABLE == 0 {
        return false;
    }
    // Skip non-image PAGE_READONLY ; it's just .rdata string literals
    // our signatures never match. Keep PAGE_READONLY when MEM_IMAGE
    // because the PE header mapping sometimes lands here.
    if info.Protect.0 == PAGE_READONLY.0 && info.Type.0 != MEM_IMAGE.0 {
        return false;
    }
    true
}

pub fn scan_all(pid: u32, pattern: &Pattern) -> io::Result<Vec<u64>> {
    let proc = Process::open(pid)?;
    let handle = proc.handle();
    let regions = scan_regions_with(handle)?;
    scan_loop(handle, &regions, pattern, true)
}

pub fn scan(pid: u32, pattern: &Pattern) -> io::Result<Option<u64>> {
    let proc = Process::open(pid)?;
    let handle = proc.handle();
    let regions = scan_regions_with(handle)?;
    if regions.is_empty() {
        return Err(io::Error::new(
            ErrorKind::NotFound,
            "no osu! code regions found; is osu! actually running?",
        ));
    }
    let hits = scan_loop(handle, &regions, pattern, false)?;
    Ok(hits.into_iter().next())
}

fn scan_regions_with(handle: HANDLE) -> io::Result<Vec<Region>> {
    let mut out = Vec::new();
    let mut addr: u64 = 0;
    let mbi_size = std::mem::size_of::<MEMORY_BASIC_INFORMATION>();
    loop {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: `&mut info` is sized correctly and `handle` is valid
        // for the caller's Process lifetime.
        let written = unsafe {
            VirtualQueryEx(handle, Some(addr as *const _), &mut info, mbi_size)
        };
        if written == 0 {
            break;
        }
        let region_start = info.BaseAddress as u64;
        let region_size = info.RegionSize as u64;
        if region_size == 0 {
            break;
        }
        let region_end = region_start.saturating_add(region_size);
        if region_kept(&info) {
            out.push(Region {
                start: region_start,
                end: region_end,
            });
        }
        if region_end <= addr {
            break;
        }
        addr = region_end;
        // osu!.exe is 32-bit; nothing we need lives above 4 GiB.
        if addr >= 0x1_0000_0000 {
            break;
        }
    }
    Ok(out)
}

fn scan_loop(
    handle: HANDLE,
    regions: &[Region],
    pattern: &Pattern,
    all: bool,
) -> io::Result<Vec<u64>> {
    const CHUNK: usize = 1 << 20; // 1 MiB ; matches linux_mem's chunking.
    let plen = pattern.len();
    let mut hits = Vec::new();
    if plen == 0 {
        return Ok(hits);
    }
    let mut buf = vec![0u8; CHUNK];
    for region in regions {
        let mut cursor = region.start;
        while cursor < region.end {
            let remaining = region.end - cursor;
            let want = CHUNK.min(remaining as usize);
            if want < plen {
                break;
            }
            let slice = &mut buf[..want];
            if read_exact_with(handle, cursor, slice).is_err() {
                // A single unreadable chunk (guard page we misfiltered,
                // torn region, …) is not fatal ; skip and keep going.
                cursor = cursor.saturating_add(want as u64);
                continue;
            }
            for i in 0..=(want - plen) {
                if pattern.matches(&slice[i..i + plen]) {
                    hits.push(cursor + i as u64);
                    if !all {
                        return Ok(hits);
                    }
                }
            }
            // Advance with plen-1 overlap so matches straddling a chunk
            // boundary aren't missed.
            cursor += (want - (plen - 1)) as u64;
        }
    }
    Ok(hits)
}
