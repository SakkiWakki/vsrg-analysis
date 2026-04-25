//! Linux-specific memory read + pattern scanning.
//!
//! ``process_vm_readv(2)`` lets a same-uid process copy pages out of
//! another's address space without attaching as a debugger. That's
//! the syscall tosu's Linux path uses and the one we use here.
//! Requires ``kernel.yama.ptrace_scope <= 1`` (the default on most
//! distros). If scanning fails with EPERM, surface a clear error
//! rather than silently returning None.

use std::fs;
use std::io::{self, ErrorKind};

use crate::pattern::Pattern;

/// Copy ``buf.len()`` bytes from ``pid``'s address space at ``addr``.
pub fn read_exact(pid: u32, addr: u64, buf: &mut [u8]) -> io::Result<()> {
    let local_iov = libc::iovec {
        iov_base: buf.as_mut_ptr() as *mut _,
        iov_len: buf.len(),
    };
    let remote_iov = libc::iovec {
        iov_base: addr as *mut _,
        iov_len: buf.len(),
    };
    // SAFETY:
    //   - local_iov.iov_base points into ``buf`` (a valid &mut [u8])
    //     and local_iov.iov_len equals buf.len(), so the kernel writes
    //     at most buf.len() bytes into memory we own. No overflow.
    //   - ``buf`` outlives this unsafe block: it's borrowed for the
    //     whole function and the borrow checker prevents drop.
    //   - remote_iov.iov_base is never dereferenced in our address
    //     space; the kernel performs the cross-process copy and
    //     returns EFAULT if the remote address is invalid.
    //   - We guard the return value below: a negative return yields
    //     an error, and a short read (impossible per man 2
    //     process_vm_readv, but defensively checked) also errors.
    let n = unsafe { libc::process_vm_readv(pid as libc::pid_t, &local_iov, 1, &remote_iov, 1, 0) };
    if n < 0 {
        return Err(io::Error::last_os_error());
    }
    // Kernel return is ssize_t; narrow to usize only after the sign
    // check so a pathological negative-cast-to-huge-usize can't slip
    // past us.
    let n = n as usize;
    if n != buf.len() {
        return Err(io::Error::new(
            ErrorKind::UnexpectedEof,
            format!("process_vm_readv short read: {n}/{}", buf.len()),
        ));
    }
    Ok(())
}

/// An address range to pattern-scan. See ``osu_scan_regions`` for
/// which mappings qualify.
struct Region {
    start: u64,
    end: u64,
}

/// Enumerate osu!'s executable memory regions for pattern scanning.
///
/// # What we keep
///
/// Every mapping that is *writable or executable* **and** either:
///   * lies inside osu!.exe's PE image (bounds derived from the PE
///     header's ``SizeOfImage``, not from maps-file pathnames ; on
///     Linux/wine the ``.text`` section is an anonymous sibling of the
///     tiny named header mapping), **or**
///   * is an anonymous ``rwxp`` region (the .NET JIT pool).
///
/// # What we drop
///
/// Named mappings to wine DLLs, shared libraries, mapped files, etc.
/// None of osu!'s signatures live there and scanning them is pure
/// waste.
///
/// # Scan order
///
/// Address-ascending (``/proc/maps`` order). The PE image lives at
/// lower addresses than the JIT pool, so signatures resident in osu!'s
/// static code are found before a same-bytes coincidence in the JIT.
///
/// # Why this is stricter than tosu
///
/// Tosu's ``tsprocess`` keeps every mapping whose perms start with
/// ``rw`` ; including all anonymous ``rw-p`` data heaps. That's
/// looser than needed (signatures never match in pure data) and it
/// drops ``r-xp`` PE code entirely. Our set is a superset of the
/// executable portion of tosu's: we find the same JIT-resident sigs
/// (e.g. ``rulesetsAddr``, ``baseAddr``) in the same address order,
/// and we additionally catch any future signature resident in PE
/// ``.text``.
fn osu_scan_regions(pid: u32) -> io::Result<Vec<Region>> {
    let maps_path = format!("/proc/{pid}/maps");
    let text = fs::read_to_string(&maps_path)?;

    // Locate osu!.exe's image base via the header mapping, then read
    // SizeOfImage out of the PE header to get the real image bounds.
    // Failure here is not fatal ; we simply scan the anonymous JIT
    // pool without the PE-image fast path.
    let image_bounds = pe_image_bounds(pid, &text).ok();

    let mut out = Vec::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let Some(range) = parts.next() else { continue };
        let Some(perms) = parts.next() else { continue };
        let (_offset, _dev, _inode) = (parts.next(), parts.next(), parts.next());
        let path = parts.next().unwrap_or("");

        let Some((s, e)) = range.split_once('-') else {
            continue;
        };
        let (Ok(start), Ok(end)) =
            (u64::from_str_radix(s, 16), u64::from_str_radix(e, 16))
        else {
            continue;
        };

        // Keep mappings inside the PE image regardless of path ;
        // wine maps the image header as named ``osu!.exe`` and the
        // rest of the image anonymously, so a path filter alone drops
        // ``.text``.
        let in_image = match image_bounds {
            Some((base, img_end)) => start >= base && end <= img_end,
            None => false,
        };
        // Otherwise only keep anonymous executable regions ; the JIT
        // pool. Named mappings (wine DLLs, .nls files, mapped .so's)
        // are not osu! code.
        let is_anon_exec = perms.contains('x') && path.is_empty();

        if !in_image && !is_anon_exec {
            continue;
        }
        out.push(Region { start, end });
    }
    Ok(out)
}

/// Resolve ``[image_base, image_base + SizeOfImage)`` for osu!.exe by
/// parsing its PE header in the target process.
///
/// Layout (little-endian):
///   [base + 0x00]: "MZ"
///   [base + 0x3c]: e_lfanew (i32, offset to PE header)
///   [base + e_lfanew]: "PE\0\0"
///   [base + e_lfanew + 0x18]: OptionalHeader magic (0x10b PE32,
///                             0x20b PE32+) ; SizeOfImage sits at
///                             +0x38 of the optional header in both.
fn pe_image_bounds(pid: u32, maps_text: &str) -> io::Result<(u64, u64)> {
    let mut image_base: Option<u64> = None;
    for line in maps_text.lines() {
        let mut parts = line.split_whitespace();
        let Some(range) = parts.next() else { continue };
        let _perms = parts.next();
        let (_offset, _dev, _inode) = (parts.next(), parts.next(), parts.next());
        let path = parts.next().unwrap_or("");
        if !path.ends_with("osu!.exe") {
            continue;
        }
        let Some((s, _)) = range.split_once('-') else { continue };
        if let Ok(start) = u64::from_str_radix(s, 16) {
            image_base = Some(start);
            break;
        }
    }
    let Some(base) = image_base else {
        return Err(io::Error::new(
            ErrorKind::NotFound,
            "osu!.exe not mapped in target process",
        ));
    };

    let mut mz = [0u8; 2];
    read_exact(pid, base, &mut mz)?;
    if &mz != b"MZ" {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "osu!.exe mapping doesn't start with MZ",
        ));
    }
    let mut e_lfanew_buf = [0u8; 4];
    read_exact(pid, base + 0x3c, &mut e_lfanew_buf)?;
    let e_lfanew = i32::from_le_bytes(e_lfanew_buf);
    if !(0..=0x1000).contains(&e_lfanew) {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("implausible e_lfanew: {e_lfanew}"),
        ));
    }
    let e_lfanew = e_lfanew as u64;
    let mut pe_sig = [0u8; 4];
    read_exact(pid, base + e_lfanew, &mut pe_sig)?;
    if &pe_sig != b"PE\0\0" {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "PE signature missing",
        ));
    }
    let optional = base + e_lfanew + 0x18;
    let mut size = [0u8; 4];
    read_exact(pid, optional + 0x38, &mut size)?;
    let size_of_image = u32::from_le_bytes(size) as u64;
    if size_of_image == 0 || size_of_image > 0x4000_0000 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("implausible SizeOfImage: {size_of_image}"),
        ));
    }
    Ok((base, base + size_of_image))
}

/// Like ``scan`` but returns every match address. Debug-only:
/// callers should prefer ``scan`` for first-match semantics.
pub fn scan_all(pid: u32, pattern: &Pattern) -> io::Result<Vec<u64>> {
    let regions = osu_scan_regions(pid)?;
    const CHUNK: usize = 1 << 20;
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
            if read_exact(pid, cursor, slice).is_err() {
                cursor = cursor.saturating_add(want as u64);
                continue;
            }
            for i in 0..=(want - plen) {
                if pattern.matches(&slice[i..i + plen]) {
                    hits.push(cursor + i as u64);
                }
            }
            cursor += (want - (plen - 1)) as u64;
        }
    }
    Ok(hits)
}

/// Scan osu!'s executable regions for ``pattern`` and return the
/// absolute address of the first match in ``/proc/maps`` address
/// order. Reads in 1 MiB chunks with a ``pattern.len() - 1``-byte
/// overlap so matches straddling a chunk boundary aren't missed.
pub fn scan(pid: u32, pattern: &Pattern) -> io::Result<Option<u64>> {
    let regions = osu_scan_regions(pid)?;
    if regions.is_empty() {
        return Err(io::Error::new(
            ErrorKind::NotFound,
            "no osu! code regions found; is osu! actually running?",
        ));
    }

    const CHUNK: usize = 1 << 20; // 1 MiB
    let plen = pattern.len();
    if plen == 0 {
        return Ok(None);
    }

    let mut buf = vec![0u8; CHUNK];
    for region in regions {
        let mut cursor = region.start;
        while cursor < region.end {
            let remaining = region.end - cursor;
            let want = CHUNK.min(remaining as usize);
            if want < plen {
                // Tail fragment smaller than the pattern; can't match.
                break;
            }
            let slice = &mut buf[..want];
            if let Err(err) = read_exact(pid, cursor, slice) {
                // One unreadable page inside a region is odd but not
                // fatal ; skip the chunk and keep going.
                cursor = cursor.saturating_add(want as u64);
                // Suppress common transient failures; surface others.
                if err.raw_os_error() != Some(libc::EIO) && err.raw_os_error() != Some(libc::EFAULT)
                {
                    // Keep moving; a single bad chunk shouldn't abort
                    // the whole scan.
                }
                continue;
            }
            for i in 0..=(want - plen) {
                if pattern.matches(&slice[i..i + plen]) {
                    return Ok(Some(cursor + i as u64));
                }
            }
            // Advance by chunk minus (plen - 1) so a match that
            // straddles the chunk boundary isn't missed.
            cursor += (want - (plen - 1)) as u64;
        }
    }
    Ok(None)
}
