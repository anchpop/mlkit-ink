/* mlkit-ink: ML Kit Digital Ink Recognition, reimplemented outside the SDK.
 *
 * Link against libmlkit_ink_ffi.a (staticlib) or libmlkit_ink_ffi.so/.dylib.
 *
 *   cargo build --release -p mlkit-ink-ffi --target aarch64-linux-android
 *   cargo build --release -p mlkit-ink-ffi --target aarch64-apple-ios
 *
 * Errors: a function that can fail returns NULL and leaves a message in
 * mlkit_ink_last_error(), which is thread-local. Read it immediately: every
 * entry point clears the slot on the way in, so the pointer is valid only
 * until the next call into this library on the same thread -- whether that
 * call fails or succeeds. Clearing eagerly means a leftover message can never
 * be read back as the explanation for an unrelated later failure.
 */
#ifndef MLKIT_INK_H
#define MLKIT_INK_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MlkitInkRecognizer MlkitInkRecognizer;

/* Load one language from raw pack bytes. The bytes are copied, so the caller
 * may free them immediately. Pass fst = NULL, fst_len = 0 to load without the
 * n-gram language model: decoding is then best-path only, which is both much
 * cheaper (the English FST is 22 MB) and, on our oracle corpus, exactly as
 * accurate. Returns NULL on failure. */
MlkitInkRecognizer *mlkit_ink_recognizer_new(const unsigned char *recospec,
                                             size_t recospec_len,
                                             const unsigned char *tflite,
                                             size_t tflite_len,
                                             const unsigned char *fst,
                                             size_t fst_len);

void mlkit_ink_recognizer_free(MlkitInkRecognizer *handle);

/* Recognize one ink. x and y hold every point of every stroke concatenated;
 * stroke_lengths gives each stroke's point count and must sum to num_points.
 * t may be NULL, in which case timestamps are synthesized at a fixed 20 ms
 * step. Pass greedy != 0 for a best-path decode.
 *
 * Returns [{"text": "...", "score": -1.23}, ...] as a NUL-terminated JSON
 * string, best first, which must be released with mlkit_ink_string_free().
 * Returns NULL on failure. */
char *mlkit_ink_recognize(const MlkitInkRecognizer *handle,
                          const double *x,
                          const double *y,
                          const double *t,
                          size_t num_points,
                          const size_t *stroke_lengths,
                          size_t num_strokes,
                          size_t nbest,
                          int greedy);

/* Switch to the empirically tuned decoder weights. They score better on our
 * corpus (72/74 against 68/74) with a sign the disassembly proves wrong, which
 * is why they are opt-in rather than the default. */
void mlkit_ink_use_empirical_weights(MlkitInkRecognizer *handle);

void mlkit_ink_string_free(char *text);

const char *mlkit_ink_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* MLKIT_INK_H */
