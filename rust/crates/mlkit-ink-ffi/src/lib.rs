//! A C ABI for `mlkit-ink`, aimed at Android (through JNI) and iOS (through Swift).
//!
//! The core crate is pure Rust with no platform dependencies, so it already
//! cross-compiles to `aarch64-linux-android` and `aarch64-apple-ios` untouched.
//! What a mobile caller actually needs is an ABI, and that is all this is.
//!
//! Three deliberate simplifications keep the surface small enough to be
//! obviously correct:
//!
//! * **Ink comes in as flat arrays** plus per-stroke lengths, so there is no
//!   struct layout to agree on across the boundary.
//! * **Results go out as a JSON string**, so there is no result type, no array
//!   of structs, and no ownership question beyond one `free`.
//! * **Errors are a thread-local string** fetched after a null return, the
//!   convention `dlerror` and friends established. It is cleared at the start
//!   of every call, so a stale message can never be mistaken for a fresh one —
//!   which means it must be read before doing anything else.
//!
//! See `include/mlkit_ink.h` for the header. Every function is safe to call
//! with a null handle; the pointer arguments must be valid for the stated
//! lengths, which is the caller's responsibility and the reason they are
//! `unsafe`.

use std::cell::RefCell;
use std::ffi::{CString, c_char, c_double, c_int};
use std::ptr;

use mlkit_ink::ink::Stroke;
use mlkit_ink::settings::DecoderSettings;
use mlkit_ink::{Recognizer as Core, Result};

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error(message: impl AsRef<str>) {
    let text = CString::new(message.as_ref()).unwrap_or_else(|_| c"error".into());
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(text));
}

fn clear_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

/// Model bytes owned here, with a recognizer borrowing from them.
///
/// The core recognizer borrows its model bytes so a desktop caller can hand it
/// an mmap for free. Across an FFI boundary there is nothing else to own them,
/// so this cell does; `self_cell` makes the self-reference safe rather than
/// this crate inventing a lifetime-erasing `unsafe` of its own.
struct Model {
    recospec: Vec<u8>,
    tflite: Vec<u8>,
    fst: Vec<u8>,
    has_fst: bool,
}

self_cell::self_cell! {
    struct Loaded {
        owner: Model,
        #[covariant]
        dependent: Core,
    }
}

/// Opaque handle. Created by [`mlkit_ink_recognizer_new`], released by
/// [`mlkit_ink_recognizer_free`].
pub struct MlkitInkRecognizer {
    loaded: Loaded,
    settings: DecoderSettings,
}

/// # Safety
/// `recospec`, `tflite` and `fst` must each point to at least the stated number
/// of bytes, or be null with a length of zero. A null `fst` loads without the
/// language model, which restricts decoding to best-path.
///
/// Returns null on failure; call [`mlkit_ink_last_error`] for the reason.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlkit_ink_recognizer_new(
    recospec: *const u8,
    recospec_len: usize,
    tflite: *const u8,
    tflite_len: usize,
    fst: *const u8,
    fst_len: usize,
) -> *mut MlkitInkRecognizer {
    clear_error();
    let Some(recospec) = (unsafe { bytes(recospec, recospec_len) }) else {
        set_error("recospec pointer is null");
        return ptr::null_mut();
    };
    let Some(tflite) = (unsafe { bytes(tflite, tflite_len) }) else {
        set_error("tflite pointer is null");
        return ptr::null_mut();
    };
    let has_fst = !fst.is_null() && fst_len > 0;
    let fst = if has_fst {
        unsafe { bytes(fst, fst_len) }.unwrap_or_default()
    } else {
        Vec::new()
    };

    let model = Model {
        recospec,
        tflite,
        fst,
        has_fst,
    };
    let loaded = Loaded::try_new(model, |model| {
        Core::load(
            &model.recospec,
            &model.tflite,
            model.has_fst.then_some(model.fst.as_slice()),
        )
    });
    match loaded {
        Ok(loaded) => {
            let settings = loaded.borrow_dependent().settings;
            Box::into_raw(Box::new(MlkitInkRecognizer { loaded, settings }))
        }
        Err(error) => {
            set_error(error.to_string());
            ptr::null_mut()
        }
    }
}

/// # Safety
/// `handle` must have come from [`mlkit_ink_recognizer_new`] and not been freed.
/// Null is accepted and ignored.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlkit_ink_recognizer_free(handle: *mut MlkitInkRecognizer) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// Recognize one ink and return `[{"text":...,"score":...}]` as JSON.
///
/// `x` and `y` hold every point of every stroke, concatenated; `stroke_lengths`
/// gives the point count of each stroke in order, and must sum to `num_points`.
/// `t` may be null, in which case timestamps are synthesized at a fixed 20 ms
/// step — that costs a little accuracy, not correctness.
///
/// Pass `greedy` nonzero for a best-path decode, which ignores the language
/// model and is the configuration that matches the real ML Kit SDK on all 74 of
/// our oracle-labelled inks.
///
/// # Safety
/// `x`, `y` and (if non-null) `t` must each be valid for `num_points` values;
/// `stroke_lengths` must be valid for `num_strokes` values.
///
/// Returns null on failure; call [`mlkit_ink_last_error`]. The returned string
/// must be released with [`mlkit_ink_string_free`].
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn mlkit_ink_recognize(
    handle: *const MlkitInkRecognizer,
    x: *const c_double,
    y: *const c_double,
    t: *const c_double,
    num_points: usize,
    stroke_lengths: *const usize,
    num_strokes: usize,
    nbest: usize,
    greedy: c_int,
) -> *mut c_char {
    clear_error();
    let Some(recognizer) = (unsafe { handle.as_ref() }) else {
        set_error("recognizer handle is null");
        return ptr::null_mut();
    };
    let strokes = match unsafe { build_strokes(x, y, t, num_points, stroke_lengths, num_strokes) } {
        Ok(strokes) => strokes,
        Err(message) => {
            set_error(message);
            return ptr::null_mut();
        }
    };

    let core = recognizer.loaded.borrow_dependent();
    let result: Result<Vec<_>> = if greedy != 0 {
        core.recognize_greedy(&strokes)
            .map(|candidate| vec![candidate])
    } else {
        core.recognize_with(&strokes, nbest.max(1), &recognizer.settings)
    };
    let candidates = match result {
        Ok(candidates) => candidates,
        Err(error) => {
            set_error(error.to_string());
            return ptr::null_mut();
        }
    };

    let payload: Vec<_> = candidates
        .iter()
        .map(|c| serde_json::json!({ "text": c.text, "score": c.score }))
        .collect();
    match CString::new(serde_json::Value::from(payload).to_string()) {
        Ok(text) => text.into_raw(),
        Err(_) => {
            set_error("recognized text contains an interior NUL");
            ptr::null_mut()
        }
    }
}

/// Switch this recognizer to the empirically tuned decoder weights. They score
/// better on our corpus (72/74 against 68/74) with a sign the disassembly
/// proves wrong, which is why they are opt-in.
///
/// # Safety
/// `handle` must have come from [`mlkit_ink_recognizer_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlkit_ink_use_empirical_weights(handle: *mut MlkitInkRecognizer) {
    if let Some(recognizer) = unsafe { handle.as_mut() } {
        let beam_width = recognizer.settings.beam_width;
        recognizer.settings = DecoderSettings {
            beam_width,
            ..DecoderSettings::empirical()
        };
    }
}

/// # Safety
/// `text` must have come from [`mlkit_ink_recognize`] and not been freed. Null
/// is accepted and ignored.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlkit_ink_string_free(text: *mut c_char) {
    if !text.is_null() {
        drop(unsafe { CString::from_raw(text) });
    }
}

/// The reason the most recent call on this thread failed, or null if it did not.
///
/// The pointer stays valid until **the next call into this library on the same
/// thread**, failing or not: every entry point clears the slot first, so that a
/// message left over from an earlier failure can never be read back as the
/// explanation for a later one. Read it immediately after a null return.
#[unsafe(no_mangle)]
pub extern "C" fn mlkit_ink_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| match slot.borrow().as_ref() {
        Some(text) => text.as_ptr(),
        None => ptr::null(),
    })
}

unsafe fn bytes(pointer: *const u8, len: usize) -> Option<Vec<u8>> {
    if pointer.is_null() {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts(pointer, len) }.to_vec())
}

unsafe fn build_strokes(
    x: *const c_double,
    y: *const c_double,
    t: *const c_double,
    num_points: usize,
    stroke_lengths: *const usize,
    num_strokes: usize,
) -> core::result::Result<Vec<Stroke>, String> {
    if x.is_null() || y.is_null() || stroke_lengths.is_null() {
        return Err(String::from("ink pointers must not be null"));
    }
    let xs = unsafe { std::slice::from_raw_parts(x, num_points) };
    let ys = unsafe { std::slice::from_raw_parts(y, num_points) };
    let ts = (!t.is_null()).then(|| unsafe { std::slice::from_raw_parts(t, num_points) });
    let lengths = unsafe { std::slice::from_raw_parts(stroke_lengths, num_strokes) };

    let total: usize = lengths.iter().sum();
    if total != num_points {
        return Err(format!(
            "stroke lengths sum to {total}, but {num_points} points were given"
        ));
    }

    let mut strokes = Vec::with_capacity(num_strokes);
    let mut start = 0;
    for &length in lengths {
        let end = start + length;
        strokes.push(Stroke {
            x: xs[start..end].to_vec(),
            y: ys[start..end].to_vec(),
            t: ts.map(|ts| ts[start..end].to_vec()).unwrap_or_default(),
            pen_up: false,
        });
        start = end;
    }
    Ok(strokes)
}
