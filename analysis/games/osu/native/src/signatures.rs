//! Named signatures and struct offsets ; the *only* place in the
//! crate that contains osu!-binary-derived magic numbers.
//!
//! # Discipline
//!
//! Everything that depends on osu!'s compiled layout lives here. Every
//! other file references these constants by name. When osu! ships a
//! binary update and a pattern breaks, the fix is a one-file change.
//!
//! # Provenance
//!
//! Patterns and offsets are ported from
//! ``tosuapp/tosu`` (MIT) at packages/tosu/src/memory/stable.ts as of
//! the 2026-04 sync. Each constant cites the tosu field it maps to so
//! future resyncs are a straightforward diff. See the repo at
//! https://github.com/tosuapp/tosu.
//!
//! # Mania-only, intentionally
//!
//! We only port what the mania live viz needs: one pointer chain into
//! the active ruleset's score data, plus struct-field offsets for the
//! headline counters. Other modes (std/taiko/catch), tourney, and
//! non-gameplay readers (chat, skin, settings) are out of scope.
//! Adding a field later is cheap; the whole point of this file is to
//! make that one-place, not sprinkle offsets across the reader.

/// A single signature: the byte pattern we scan for, plus a signed
/// offset added to the match address before use.
///
/// ``offset_from_match`` is what tosu calls ``offset``. Negative
/// offsets are legitimate ; some patterns match an instruction a few
/// bytes *after* the pointer we actually want.
#[derive(Debug, Clone, Copy)]
pub struct Signature {
    pub name: &'static str,
    pub pattern: &'static str,
    pub offset_from_match: i64,
}

// ─── Signatures ───────────────────────────────────────────────────────────

/// Entry point into the gameplay pointer chain. Walking it:
///
/// ```text
/// ruleset      = read_u32(read_u32(RULESETS_PTR.addr + RULESETS_IND_OFFSET) + 0x4)
/// gameplayBase = read_u32(ruleset + RULESET_GAMEPLAY_OFFSET)
/// scoreBase    = read_u32(gameplayBase + GAMEPLAY_SCORE_OFFSET)
/// ```
///
/// The ``-0xb`` and ``+0x4`` constants below are the two immediate
/// dereferences tosu does on the pattern match (not raw struct
/// offsets ; they're part of the instructions the pattern matched
/// against, so they live here with the signature).
///
/// Source: tosu StableMemory.scanPatterns.rulesetsAddr.
pub const RULESETS_PTR: Signature = Signature {
    name: "rulesets_ptr",
    pattern: "7D 15 A1 ?? ?? ?? ?? 85 C0",
    offset_from_match: 0,
};

/// Immediate to add to the ``RULESETS_PTR`` match before the first
/// deref. Part of the mov instruction the pattern matches, not a
/// struct layout offset ; but keeping it here avoids magic numbers in
/// the reader.
pub const RULESETS_IND_OFFSET: i64 = -0xb;

/// Second immediate in the ruleset indirection: pointer in the vtable
/// slot 4 bytes past the indirected address.
pub const RULESETS_VTABLE_OFFSET: u64 = 0x4;

/// Anchor for the beatmap pointer chain. Walking it:
///
/// ```text
/// slot    = read_u32(BASE_ADDR.addr + BASE_TO_BEATMAP_OFFSET)
/// beatmap = read_u32(slot)
/// ```
///
/// Source: tosu StableMemory.scanPatterns.baseAddr.
pub const BASE_ADDR: Signature = Signature {
    name: "base_addr",
    pattern: "F8 01 74 04 83 65",
    offset_from_match: 0,
};

/// Offset from ``BASE_ADDR``'s match to the 4-byte immediate/static
/// field slot used by the current beatmap pointer chain. Tosu calls
/// ``readPointer(baseAddr - 0xc)``; its ``readPointer`` first reads
/// this slot address from the JIT instruction stream, then reads the
/// beatmap struct pointer stored at that slot.
pub const BASE_TO_BEATMAP_OFFSET: i64 = -0xc;

/// Entry point into the game-state enum. Walking it mirrors
/// ``BASE_ADDR``:
///
/// ```text
/// slot   = read_u32(STATUS_PTR.addr + STATUS_IND_OFFSET)
/// status = read_u32(slot)
/// ```
///
/// ``status`` is an ``osu::GameState`` enum value; compare against
/// ``GAME_STATE_PLAY`` to detect "actually in gameplay" vs menu /
/// results / song select / etc. The overlay uses this to hide itself
/// whenever the user isn't actively playing.
///
/// Source: tosu StableMemory.scanPatterns.statusPtr.
pub const STATUS_PTR: Signature = Signature {
    name: "status_ptr",
    pattern: "48 83 F8 04 73 1E",
    offset_from_match: 0,
};

/// Immediate applied to the ``STATUS_PTR`` match before the first
/// deref. Tosu ships this as the signature's ``offset`` field.
pub const STATUS_IND_OFFSET: i64 = -0x4;

/// Value of ``status`` when the player is in active gameplay.
/// Matches ``GameState.play`` in tosu's enum.
pub const GAME_STATE_PLAY: u32 = 2;

// ─── Struct field offsets ─────────────────────────────────────────────────
//
// Offsets into the structs reached by the pointer chain above.
// Grouped by their parent struct so it's obvious which pointer to add
// them to.

/// Offsets into the ``ruleset`` struct (first deref past rulesets).
pub mod ruleset {
    /// Pointer to the gameplay state. ``[ruleset + 0x68]`` in tosu.
    pub const GAMEPLAY_PTR: u64 = 0x68;
}

/// Offsets into the ``gameplayBase`` struct.
pub mod gameplay {
    /// Pointer to the current score object. ``[gameplay + 0x38]`` in tosu.
    pub const SCORE_PTR: u64 = 0x38;
    /// Pointer to the accuracy wrapper. ``[gameplay + 0x48]``; the
    /// accuracy double lives at ``+0xC`` past that pointer (see
    /// ``ACCURACY_OFFSET`` below).
    pub const ACCURACY_PTR: u64 = 0x48;
}

/// Offsets into the ``scoreBase`` struct. Each field is a u16 (short)
/// in osu!'s score object.
pub mod score {
    pub const HIT_100: u64 = 0x88;
    pub const HIT_300: u64 = 0x8a;
    pub const HIT_50: u64 = 0x8c;
    pub const HIT_GEKI: u64 = 0x8e;
    pub const HIT_KATU: u64 = 0x90;
    pub const HIT_MISS: u64 = 0x92;
    pub const COMBO: u64 = 0x94;
    pub const MAX_COMBO: u64 = 0x68;
    pub const MODE: u64 = 0x64;

    /// Pointer to the hit-error ``List<int>`` wrapper. Walking it:
    /// ``[[scoreBase + HIT_ERRORS_LIST_PTR] + LIST_ITEMS_PTR] + LEADER_START + 4*i``.
    pub const HIT_ERRORS_LIST_PTR: u64 = 0x38;
}

/// Offsets inside the ``accuracy_wrapper`` reached via
/// ``gameplay::ACCURACY_PTR``.
pub mod accuracy {
    /// Double-precision accuracy at +0xC into the wrapper.
    pub const VALUE: u64 = 0xc;
}

/// Offsets into the beatmap struct (reached via ``BASE_ADDR``).
///
/// All string fields here are pointers to .NET ``System.String``
/// objects; use ``dotnet_string`` below to decode them. The ``cs``
/// field is a 32-bit float (mania keycount lives here for mania maps).
#[allow(dead_code)]
pub mod beatmap {
    /// Pointer to the md5 checksum string.
    pub const MD5: u64 = 0x6c;
    /// Pointer to the .osu filename string.
    pub const FILENAME: u64 = 0x90;

    /// Metadata string pointers (all .NET System.String refs; decode
    /// with ``dotnet_string``).
    pub const ARTIST: u64 = 0x18;
    pub const ARTIST_UNICODE: u64 = 0x1c;
    /// Romanized title. tosu calls this ``title``; the original
    /// (CJK/etc.) string is at ``TITLE_UNICODE``.
    pub const TITLE: u64 = 0x24;
    pub const TITLE_UNICODE: u64 = 0x28;
    pub const AUDIO_FILENAME: u64 = 0x64;
    pub const BACKGROUND_FILENAME: u64 = 0x68;
    pub const FOLDER: u64 = 0x78;
    pub const CREATOR: u64 = 0x7c;
    pub const VERSION: u64 = 0xac;

    /// Float difficulty stats (single-precision). Order in memory is
    /// AR, CS, HP, OD at consecutive +0x4 slots starting at +0x2c.
    pub const AR: u64 = 0x2c;
    pub const CS: u64 = 0x30;
    pub const HP: u64 = 0x34;
    pub const OD: u64 = 0x38;

    /// Integer fields.
    pub const BEATMAP_ID: u64 = 0xc8;
    pub const BEATMAP_SET_ID: u64 = 0xcc;
    pub const OBJECT_COUNT: u64 = 0xf8;
    pub const RANKED_STATUS: u64 = 0x12c;
}

/// Offsets inside a .NET ``System.String`` (32-bit CLR).
///
/// Layout: ``[addr + 0]`` is the method-table pointer (ignored),
/// ``[addr + 4]`` is an int32 character count, and
/// ``[addr + 8 ... + 8 + 2*length]`` is UTF-16LE data. Tosu caps
/// length at 4096 to reject obviously-bogus pointers; we do the
/// same.
pub mod dotnet_string {
    pub const LENGTH_OFFSET: u64 = 0x4;
    pub const DATA_OFFSET: u64 = 0x8;
    /// Sanity cap on character count. A real string longer than this
    /// almost certainly means the pointer is pointing at random
    /// memory; reading ahead would waste time and risk EFAULT.
    pub const MAX_LENGTH: usize = 4096;
}

/// Offsets inside a .NET ``List<T>`` backing-array header.
///
/// osu! is a .NET 4 app; its ``List<T>`` has an ``_items`` ``T[]``
/// reference at +0x4 and a ``_size`` int at +0xC. The array itself
/// stores elements starting at ``LEADER_START`` past the header
/// (``_items`` points at the array object, not the payload).
pub mod dotnet_list {
    pub const ITEMS_PTR: u64 = 0x4;
    pub const SIZE: u64 = 0xc;
    /// Offset from the start of a T[] object to element 0. The
    /// .NET array object header is 8 bytes on x86 (sync block +
    /// method table); after that, elements of a primitive T[] begin.
    pub const LEADER_START: u64 = 0x8;
}
