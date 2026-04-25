//! High-level reader: resolve signatures once, then read mania
//! gameplay state on demand.
//!
//! # Design
//!
//! Two layers. **Resolution** scans the process memory for every
//! signature in ``signatures`` and caches the resulting absolute
//! addresses in a ``Resolved`` handle. **Reading** walks the pointer
//! chain from that handle to the live gameplay struct and copies out
//! the fields we care about.
//!
//! The pointer chain (``rulesets → ruleset → gameplay → score``) is
//! walked in exactly one place ; ``gameplay_pointers`` ; and every
//! reader consumes its output. Compare to tosu's per-method
//! copy-paste.
//!
//! # Mania-only
//!
//! All reads assume the active ruleset is mania. Calling these
//! functions while the player is in std/taiko/catch will still
//! succeed ; the struct layout is shared ; but the numbers may or
//! may not be meaningful. The caller should gate reads on the
//! plugin's existing mania check.

use std::io;

use crate::mem;
use crate::pattern;
use crate::signatures::{
    self as sig, accuracy, beatmap, dotnet_list, dotnet_string, gameplay, ruleset, score, Signature,
    GAME_STATE_PLAY,
};

/// Cached signature resolutions for one osu! process. Cheap to clone
/// (all fields are copyable integers). Invalidated by an osu!
/// restart ; the PID and absolute addresses both change.
#[derive(Debug, Clone, Copy)]
pub struct Resolved {
    pub pid: u32,
    /// Absolute address of the match for ``RULESETS_PTR``.
    pub rulesets_ptr: u64,
    /// Absolute address of the match for ``BASE_ADDR``.
    pub base_addr: u64,
    /// Absolute address of the match for ``STATUS_PTR``.
    pub status_ptr: u64,
}

/// Scan the process once and return a handle with every known
/// signature resolved. Fails if any required pattern is missing ;
/// better to fail loudly than let the reader return zeros.
pub fn resolve(pid: u32) -> io::Result<Resolved> {
    let rulesets_ptr = scan_required(pid, &sig::RULESETS_PTR)?;
    let base_addr = scan_required(pid, &sig::BASE_ADDR)?;
    let status_ptr = scan_required(pid, &sig::STATUS_PTR)?;
    Ok(Resolved {
        pid,
        rulesets_ptr,
        base_addr,
        status_ptr,
    })
}

fn scan_required(pid: u32, s: &Signature) -> io::Result<u64> {
    let pat = pattern::parse(s.pattern).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("signature {:?} has invalid pattern: {e}", s.name),
        )
    })?;
    let hit = mem::scan(pid, &pat)?;
    let Some(addr) = hit else {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "signature {:?} not found in tosu-compatible scan regions; osu! \
                 may have \
                 been updated and the pattern in signatures.rs needs \
                 re-deriving",
                s.name
            ),
        ));
    };
    // Apply the signed offset to the match. ``as u64`` is fine even
    // for negative offsets because the underlying arithmetic is
    // wrapping ; addresses are unsigned 64-bit, and the pattern
    // authors chose offsets knowing the match site.
    Ok(addr.wrapping_add(s.offset_from_match as u64))
}

/// Pointers reached by walking the mania gameplay chain. Every field
/// that *can* be zero when the player is on the menu is wrapped in
/// ``Option`` so the Python side can distinguish "not playing" from
/// "playing with combo == 0".
#[derive(Debug, Clone, Copy, Default)]
pub struct GameplayPointers {
    pub ruleset: Option<u64>,
    pub gameplay: Option<u64>,
    pub score: Option<u64>,
    pub accuracy_wrapper: Option<u64>,
}

/// Walk the rulesets → ruleset → gameplay → score pointer chain.
///
/// This is the single source of truth for the chain. Anything needing
/// any rung of it calls this once and picks the field it wants from
/// the returned struct. Tosu calls it inline in four separate reader
/// methods; we don't.
///
/// Returns ``Ok`` even when the chain hits a null pointer (means
/// "not currently playing") ; callers branch on ``Option`` emptiness.
/// ``Err`` is reserved for syscall failures.
pub fn gameplay_pointers(r: &Resolved) -> io::Result<GameplayPointers> {
    let mut out = GameplayPointers::default();

    let base = (r.rulesets_ptr as i64 + sig::RULESETS_IND_OFFSET) as u64;
    let indirected = read_u32_as_u64(r.pid, base)?;
    if indirected == 0 {
        return Ok(out);
    }
    let ruleset = read_u32_as_u64(r.pid, indirected + sig::RULESETS_VTABLE_OFFSET)?;
    if ruleset == 0 {
        return Ok(out);
    }
    out.ruleset = Some(ruleset);

    let gameplay = read_u32_as_u64(r.pid, ruleset + ruleset::GAMEPLAY_PTR)?;
    if gameplay == 0 {
        return Ok(out);
    }
    out.gameplay = Some(gameplay);

    let score = read_u32_as_u64(r.pid, gameplay + gameplay::SCORE_PTR)?;
    if score != 0 {
        out.score = Some(score);
    }

    let accuracy_wrapper = read_u32_as_u64(r.pid, gameplay + gameplay::ACCURACY_PTR)?;
    if accuracy_wrapper != 0 {
        out.accuracy_wrapper = Some(accuracy_wrapper);
    }

    Ok(out)
}

/// Mania gameplay state. Fields are ``Option`` where null-in-osu
/// means "not playing"; the scalar fields (``combo`` etc.) are zero
/// when not playing, matching osu!'s in-memory defaults.
///
/// Shape mirrors the Python-side ``GameMemoryState`` / ``ChartMetadata``
/// / ``ChartStats`` split: per-play counters stay on the top level;
/// chart identity and difficulty stats live in ``chart_meta`` and
/// ``chart_stats`` bundles that are populated independently of the
/// gameplay pointer chain.
#[derive(Debug, Clone, Default)]
pub struct ManiaState {
    pub playing: bool,
    pub combo: u16,
    pub max_combo: u16,
    pub mode: u32,
    pub accuracy: f64,

    /// Per-play judgment counters. Kept individually (not a hashmap)
    /// because the Rust side is a pure C-ABI shuttle; the Python
    /// wrapper repackages these into a judgment_counts dict for
    /// GameMemoryState.
    pub hit_300: u16,
    pub hit_100: u16,
    pub hit_50: u16,
    pub hit_geki: u16,
    pub hit_katu: u16,
    pub hit_miss: u16,

    /// Hit-error offsets in ms. Tosu clamps to [-10_000, 10_000]; we
    /// apply the same clamp to reject obvious garbage from stale
    /// pointers.
    pub hit_errors_ms: Vec<i32>,

    /// Chart identity + difficulty, populated whenever the beatmap
    /// pointer is live (works on song select too, not just gameplay).
    pub chart_meta: ChartMeta,
    pub chart_stats: ChartStats,

    /// Raw ``osu::GameState`` enum value. 2 == play (actively in
    /// gameplay); other values indicate menu, results, song select,
    /// etc. Exposed as-is so callers can filter however they want.
    pub game_state: u32,
    /// Convenience flag: ``game_state == GAME_STATE_PLAY``.
    pub in_gameplay: bool,
}

/// Chart identity strings. Empty strings mean the beatmap pointer
/// was null or the field was unreadable. Mirrors Python's
/// ``ChartMetadata``.
#[derive(Debug, Clone, Default)]
pub struct ChartMeta {
    pub md5: String,
    pub filename: String,
    pub artist: String,
    pub artist_unicode: String,
    pub title: String,
    pub title_unicode: String,
    pub creator: String,
    pub version: String,
    pub audio_filename: String,
    pub background_filename: String,
    pub folder: String,
    pub beatmap_id: u32,
    pub beatmap_set_id: u32,
    pub ranked_status: u32,
}

/// Chart difficulty stats. Mirrors Python's ``ChartStats`` but with
/// osu's native fields kept individually (AR/CS/HP/OD); the Python
/// side stashes them under ``ChartStats.extra`` to keep the dataclass
/// game-agnostic.
#[derive(Debug, Clone, Default)]
pub struct ChartStats {
    pub ar: f32,
    pub cs: f32,
    pub hp: f32,
    pub od: f32,
    pub object_count: u32,
}

pub fn read_mania_state(r: &Resolved) -> io::Result<ManiaState> {
    let mut out = ManiaState::default();

    // Game state is cheap (two reads) and needed on every tick so
    // the overlay can show/hide without touching the gameplay chain.
    out.game_state = read_game_state(r).unwrap_or(0);
    out.in_gameplay = out.game_state == GAME_STATE_PLAY;

    // Beatmap data is readable independently of the gameplay chain
    // (the player can be on the song select screen with a map
    // highlighted but no active ruleset), so read it first and
    // unconditionally. A null beatmap pointer leaves the defaults.
    if let Some(bm_base) = read_beatmap_ptr(r)? {
        out.chart_meta = read_chart_meta(r.pid, bm_base)?;
        out.chart_stats = read_chart_stats(r.pid, bm_base)?;
    }

    let ptrs = gameplay_pointers(r)?;
    let (Some(score_base), Some(acc_wrapper)) = (ptrs.score, ptrs.accuracy_wrapper) else {
        return Ok(out);
    };

    out.playing = true;
    out.combo = read_u16(r.pid, score_base + score::COMBO)?;
    out.max_combo = read_u16(r.pid, score_base + score::MAX_COMBO)?;
    out.mode = read_u32(r.pid, score_base + score::MODE)?;
    out.accuracy = read_f64(r.pid, acc_wrapper + accuracy::VALUE)?;
    out.hit_300 = read_u16(r.pid, score_base + score::HIT_300)?;
    out.hit_100 = read_u16(r.pid, score_base + score::HIT_100)?;
    out.hit_50 = read_u16(r.pid, score_base + score::HIT_50)?;
    out.hit_geki = read_u16(r.pid, score_base + score::HIT_GEKI)?;
    out.hit_katu = read_u16(r.pid, score_base + score::HIT_KATU)?;
    out.hit_miss = read_u16(r.pid, score_base + score::HIT_MISS)?;

    out.hit_errors_ms = read_hit_errors(r.pid, score_base)?;
    Ok(out)
}

/// Read the current ``osu::GameState`` enum value. Mirrors
/// ``BASE_ADDR``'s double-deref pattern: the ``STATUS_PTR`` match is
/// inside a JIT ``cmp`` instruction whose immediate at
/// ``STATUS_IND_OFFSET`` is the static field slot; one more deref
/// yields the u32 value.
fn read_game_state(r: &Resolved) -> io::Result<u32> {
    let addr = (r.status_ptr as i64 + sig::STATUS_IND_OFFSET) as u64;
    let slot = read_u32_as_u64(r.pid, addr)?;
    if slot == 0 {
        return Ok(0);
    }
    read_u32(r.pid, slot)
}

/// Dereference ``BASE_ADDR`` to the current beatmap struct.
///
/// Tosu's TypeScript ``readPointer(addr)`` is a double dereference:
/// ``readIntPtr(readIntPtr(addr))``. For ``baseAddr - 0xc`` the first
/// read pulls the static field slot address out of the JIT instruction
/// stream; the second read pulls the current beatmap object pointer
/// from that slot. Returns ``None`` if either rung is null.
fn read_beatmap_ptr(r: &Resolved) -> io::Result<Option<u64>> {
    let addr = (r.base_addr as i64 + sig::BASE_TO_BEATMAP_OFFSET) as u64;
    let slot = read_u32_as_u64(r.pid, addr)?;
    if slot == 0 {
        return Ok(None);
    }
    let bm = read_u32_as_u64(r.pid, slot)?;
    Ok(if bm == 0 { None } else { Some(bm) })
}

/// Populate chart identity strings + IDs from the beatmap struct.
///
/// Each field is read independently; a failure on any one doesn't
/// abort the read (beatmaps sometimes have null pointers for optional
/// fields like ``title_unicode`` on ASCII-only charts). Errors from
/// the underlying ``read_string_at_ptr`` propagate -- those indicate
/// genuine memory-access failures, not missing optional fields.
fn read_chart_meta(pid: u32, bm: u64) -> io::Result<ChartMeta> {
    Ok(ChartMeta {
        md5:                 read_string_at_ptr(pid, bm + beatmap::MD5)?,
        filename:            read_string_at_ptr(pid, bm + beatmap::FILENAME)?,
        artist:              read_string_at_ptr(pid, bm + beatmap::ARTIST)?,
        artist_unicode:      read_string_at_ptr(pid, bm + beatmap::ARTIST_UNICODE)?,
        title:               read_string_at_ptr(pid, bm + beatmap::TITLE)?,
        title_unicode:       read_string_at_ptr(pid, bm + beatmap::TITLE_UNICODE)?,
        creator:             read_string_at_ptr(pid, bm + beatmap::CREATOR)?,
        version:             read_string_at_ptr(pid, bm + beatmap::VERSION)?,
        audio_filename:      read_string_at_ptr(pid, bm + beatmap::AUDIO_FILENAME)?,
        background_filename: read_string_at_ptr(pid, bm + beatmap::BACKGROUND_FILENAME)?,
        folder:              read_string_at_ptr(pid, bm + beatmap::FOLDER)?,
        beatmap_id:          read_u32(pid, bm + beatmap::BEATMAP_ID)?,
        beatmap_set_id:      read_u32(pid, bm + beatmap::BEATMAP_SET_ID)?,
        ranked_status:       read_u32(pid, bm + beatmap::RANKED_STATUS)?,
    })
}

fn read_chart_stats(pid: u32, bm: u64) -> io::Result<ChartStats> {
    Ok(ChartStats {
        ar:           read_f32(pid, bm + beatmap::AR)?,
        cs:           read_f32(pid, bm + beatmap::CS)?,
        hp:           read_f32(pid, bm + beatmap::HP)?,
        od:           read_f32(pid, bm + beatmap::OD)?,
        object_count: read_u32(pid, bm + beatmap::OBJECT_COUNT)?,
    })
}

/// Given a pointer-to-a-pointer-to-a-.NET-string, read the string.
/// Returns an empty string if the indirection is null ; the caller
/// can't distinguish "no string" from "empty string" but in osu!'s
/// data model the fields we read are never legitimately "".
fn read_string_at_ptr(pid: u32, addr: u64) -> io::Result<String> {
    let string_obj = read_u32_as_u64(pid, addr)?;
    if string_obj == 0 {
        return Ok(String::new());
    }
    read_dotnet_string(pid, string_obj)
}

/// Decode a .NET ``System.String`` at ``addr``. See
/// ``sig::dotnet_string`` for the layout.
fn read_dotnet_string(pid: u32, addr: u64) -> io::Result<String> {
    let length = read_u32(pid, addr + dotnet_string::LENGTH_OFFSET)?;
    // Reject obviously-bogus lengths. A real string hitting this cap
    // is vanishingly unlikely; a stale pointer hitting it is common.
    if length == 0 || (length as usize) >= dotnet_string::MAX_LENGTH {
        return Ok(String::new());
    }
    let byte_len = (length as usize) * 2;
    let mut buf = vec![0u8; byte_len];
    mem::read_exact(pid, addr + dotnet_string::DATA_OFFSET, &mut buf)?;
    // .NET strings are UTF-16LE. Decode into a Vec<u16> first; use
    // String::from_utf16_lossy so malformed surrogates don't abort
    // the whole read (garbage pointers sometimes hit valid memory
    // that isn't real UTF-16).
    let u16s: Vec<u16> = buf
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    Ok(String::from_utf16_lossy(&u16s))
}

/// Read the ``List<int> _hitErrors`` backing the hit-error array.
/// Layout: ``[[scoreBase + HIT_ERRORS_LIST_PTR] + ITEMS_PTR]`` is the
/// ``int[]`` object; elements start at ``LEADER_START`` past it.
fn read_hit_errors(pid: u32, score_base: u64) -> io::Result<Vec<i32>> {
    let list = read_u32_as_u64(pid, score_base + score::HIT_ERRORS_LIST_PTR)?;
    if list == 0 {
        return Ok(Vec::new());
    }
    let items = read_u32_as_u64(pid, list + dotnet_list::ITEMS_PTR)?;
    let size = read_u32(pid, list + dotnet_list::SIZE)?;
    if items == 0 || size == 0 {
        return Ok(Vec::new());
    }
    // Reject implausible sizes ; a bogus pointer chain can return
    // 0xFFFFFFFF here and we'd try to allocate 16 GB.
    const MAX_HITS: u32 = 1_000_000;
    if size > MAX_HITS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "hit-error list size {size} exceeds {MAX_HITS}; \
                 pointer chain likely stale"
            ),
        ));
    }

    let mut out = Vec::with_capacity(size as usize);
    // Read the whole array in one syscall rather than one per
    // element. Each entry is an i32.
    let bytes_len = (size as usize) * 4;
    let mut buf = vec![0u8; bytes_len];
    mem::read_exact(pid, items + dotnet_list::LEADER_START, &mut buf)?;
    for chunk in buf.chunks_exact(4) {
        let v = i32::from_le_bytes(chunk.try_into().unwrap());
        // Same clamp tosu applies ; stale chains sometimes surface
        // million-ish values.
        if v < -10_000 || v > 10_000 {
            break;
        }
        out.push(v);
    }
    Ok(out)
}

// ─── Typed read helpers ───────────────────────────────────────────────────

fn read_u32(pid: u32, addr: u64) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    mem::read_exact(pid, addr, &mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u32_as_u64(pid: u32, addr: u64) -> io::Result<u64> {
    Ok(read_u32(pid, addr)? as u64)
}

fn read_u16(pid: u32, addr: u64) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    mem::read_exact(pid, addr, &mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_f64(pid: u32, addr: u64) -> io::Result<f64> {
    let mut buf = [0u8; 8];
    mem::read_exact(pid, addr, &mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_f32(pid: u32, addr: u64) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    mem::read_exact(pid, addr, &mut buf)?;
    Ok(f32::from_le_bytes(buf))
}
