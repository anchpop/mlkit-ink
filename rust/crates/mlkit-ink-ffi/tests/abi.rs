//! Exercise the C ABI from Rust, through the same raw pointers a JNI or Swift
//! caller would pass.
//!
//! A C header that compiles proves nothing about whether the ABI works; these
//! call the exported symbols with the exact pointer/length conventions the
//! header documents, including the failure paths, which are the ones a mobile
//! caller is most likely to hit first.

use std::ffi::{CStr, c_char};
use std::path::PathBuf;

use mlkit_ink_ffi::{
    mlkit_ink_last_error, mlkit_ink_recognize, mlkit_ink_recognizer_free, mlkit_ink_recognizer_new,
    mlkit_ink_string_free,
};

fn repo_root() -> PathBuf {
    match std::env::var_os("MLKIT_INK_ROOT") {
        Some(dir) => PathBuf::from(dir),
        None => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .canonicalize()
            .unwrap(),
    }
}

const RECOSPEC: &str = "models/qrnn_en_us_reco_20200318_fst_20191208_recospec_zip/\
                        qrnn.en_us.reco_20200318.fst_20191208.recospec.local";
const TFLITE: &str =
    "models/indy_lstm_latin_6x216_tflite_20191208_zip/latin_indy_lstm_6x216_20191208.tflite";

fn last_error() -> String {
    let pointer = mlkit_ink_last_error();
    if pointer.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}

fn take_string(pointer: *mut c_char) -> String {
    assert!(!pointer.is_null(), "call failed: {}", last_error());
    let text = unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned();
    unsafe { mlkit_ink_string_free(pointer) };
    text
}

#[test]
fn recognizes_an_ink_through_the_c_abi() {
    let root = repo_root();
    let (Ok(recospec), Ok(tflite)) = (
        std::fs::read(root.join(RECOSPEC)),
        std::fs::read(root.join(TFLITE)),
    ) else {
        eprintln!("skipping: en-US packs are not fetched; run `mlkit-ink fetch en-US`");
        return;
    };

    let handle = unsafe {
        mlkit_ink_recognizer_new(
            recospec.as_ptr(),
            recospec.len(),
            tflite.as_ptr(),
            tflite.len(),
            std::ptr::null(),
            0,
        )
    };
    assert!(!handle.is_null(), "load failed: {}", last_error());

    let ink = std::fs::read_to_string(root.join("testdata/corpus/trials/reference-hi/ink.json"))
        .expect("trial ink");
    let (x, y, t, lengths) = flatten(&ink);

    let json = take_string(unsafe {
        mlkit_ink_recognize(
            handle,
            x.as_ptr(),
            y.as_ptr(),
            t.as_ptr(),
            x.len(),
            lengths.as_ptr(),
            lengths.len(),
            1,
            1,
        )
    });
    assert!(json.contains("\"hi\""), "unexpected result: {json}");

    unsafe { mlkit_ink_recognizer_free(handle) };
}

#[test]
fn null_and_mismatched_inputs_report_errors_instead_of_crashing() {
    let handle = unsafe {
        mlkit_ink_recognizer_new(
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        )
    };
    assert!(handle.is_null());
    assert!(last_error().contains("recospec"), "{}", last_error());

    // A null handle must be rejected, not dereferenced.
    let result = unsafe {
        mlkit_ink_recognize(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
            1,
            1,
        )
    };
    assert!(result.is_null());
    assert!(last_error().contains("handle"), "{}", last_error());

    // Freeing null is a no-op, as the header promises.
    unsafe { mlkit_ink_recognizer_free(std::ptr::null_mut()) };
    unsafe { mlkit_ink_string_free(std::ptr::null_mut()) };
}

/// Ink JSON -> the flat-array convention the ABI uses.
fn flatten(json: &str) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>) {
    let value: serde_json::Value = serde_json::from_str(json).expect("ink JSON");
    let (mut x, mut y, mut t, mut lengths) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for stroke in value["strokes"].as_array().expect("strokes") {
        let read = |key: &str| -> Vec<f64> {
            stroke[key]
                .as_array()
                .map(|items| items.iter().map(|v| v.as_f64().expect("number")).collect())
                .unwrap_or_default()
        };
        let (sx, sy, st) = (read("x"), read("y"), read("t"));
        lengths.push(sx.len());
        x.extend(sx);
        y.extend(sy);
        t.extend(st);
    }
    (x, y, t, lengths)
}
