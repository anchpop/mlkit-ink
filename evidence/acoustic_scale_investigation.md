# `acoustic_scale` investigation

Binary: `android-oracle/app/build/intermediates/stripped_native_libs/debug/stripDebugDebugSymbols/out/lib/arm64-v8a/libdigitalink.so`
(md5 `600612f063fc34ea00171f8290d17187`; identical to the `merged_native_libs` copy). All addresses below
are file offsets == vaddrs (first `PT_LOAD`: `off 0x0 vaddr 0x0`, `filesz 0x656e00`), confirmed against the
already-verified SPEC.md §18 site `0x4ea5dc` (options-transfer bitfield copy — disassembly at that exact
address matches SPEC's description, see "Sanity check" below).

Tools used: `objdump` (Apple's bundled LLVM objdump handles `elf64-littleaarch64` fine), `radare2 6.1.0`
(disassembly only, no analysis passes — `aaa` was not run, all addressing is manual), and Python for
reading raw ELF bytes / `.rela.dyn` (`R_AARCH64_RELATIVE`, type 1027) relocation entries to resolve C++
RTTI (typeinfo names → typeinfo objects → vtables → virtual function addresses) without needing a
disassembler's own symbolication.

## Summary of findings

- **The task's hint address (`0x4f1182..0x4f118c`) does NOT hold the acoustic_scale computation.**
  That address range decodes (at the nearest valid instruction boundaries, `0x4f1180`/`0x4f1184`/`0x4f1188`/`0x4f118c`)
  to a repeated `bl 0x4f5060` / `bl 0x4f5030` / `b <target>` / `bl 0x4f5238` pattern with no floating-point
  instructions anywhere in `0x4f0000..0x4f2000` except one unrelated `fmov` and two unrelated `ldr s/d`
  hundreds of bytes away. This is the classic shape of LLVM/gcc `.gcc_except_table`-driven C++ exception
  **cleanup landing pads** (call a couple of local-variable destructors, then resume unwinding at a
  different point) — the binary does have a populated `.gcc_except_table` section, consistent with this
  read. I found the real computation independently via RTTI (below); it is **not** at this address in this
  build of the library. I flag this as a genuine discrepancy with SPEC.md, not something I can reconcile —
  possibly SPEC's investigator used a different tool's address numbering, or made a transcription slip; I
  cannot tell which from a single build.

- **The real computation, found via RTTI, IS exactly SPEC's formula** `cost = -acoustic_scale * (first_frame_max [+ f10])`,
  located at `0x4152b0..0x4152dc`, inside `research_handwriting::PreSpaceAwareNetworkScoreCache`'s override
  of a virtual scoring method.

- **`acoustic_scale` is a per-instance runtime field** (stored pre-negated), threaded into the object via a
  float constructor argument, not a compile-time immediate. I traced the *entire* single code path that
  supplies this argument, down to a `ldr s8, [x22, #0x90]` read from a `research_handwriting::FstDecoder`
  instance (confirmed via RTTI/vtable). **I could not find the instruction that originally writes that
  field** (see "What I could not determine"), so **the numeric value of `acoustic_scale` for en-US is
  UNDETERMINED** from this investigation, despite an extensive search.

- **Q3 (the question that matters) has a definitive, address-anchored answer: YES.** The base class
  `research_handwriting::NetworkScoreCache`'s own (non-overridden) per-frame/per-label score-lookup method,
  at `0x42ed14`, multiplies the raw neural-net log-posterior read straight out of the net-output array by
  the *same* cached field (`-acoustic_scale`) before returning it (`0x42ed74..0x42ed88`). This is the
  fundamental acoustic-score accessor used throughout decoding — every acoustic score the decoder ever
  sees for a normal (non-synthetic-space) frame/label pair is `-acoustic_scale * raw_net_logit`. If
  `acoustic_scale != 1.0`, **every** acoustic score is scaled by that factor before being combined with the
  (unscaled) LM/FST costs from §17, exactly as SPEC.md §19 worried.

## 1. Locating `PreSpaceAwareNetworkScoreCache` and its scoring method (proves the formula, not the value)

RTTI name string `N20research_handwriting30PreSpaceAwareNetworkScoreCacheE` is at file offset `0xade6a`.
Standard Itanium ABI chase, done by resolving `R_AARCH64_RELATIVE` addends in `.rela.dyn` (base `0x2268`,
size `0x78960`, 24-byte `Elf64_Rela` entries: `r_offset, r_info, r_addend`):

- typeinfo-name pointer field written by a relocation with `r_addend = 0xade6a` at `r_offset = 0x65f9b8`
  → typeinfo object at `0x65f9b0`.
- typeinfo-object pointer referenced by exactly one relocation, `r_addend = 0x65f9b0` at
  `r_offset = 0x65f948` → this is `vtable + 16` (the typeinfo slot), so **vtable base = `0x65f940`**,
  first virtual function at `0x65f950`.

Vtable dump (each `R_AARCH64_RELATIVE` addend at `vtable+16+8i`):

```
0x65f950 -> 0x1af358   (shared dtor stub, reused by other classes too)
0x65f958 -> 0x1af488   (deleting dtor)
0x65f960 -> 0x1af358
0x65f968 -> 0x415298   <-- scoring method override (see below)
0x65f970 -> 0x4152ec
0x65f978 -> 0x41531c
0x65f980 -> 0x415350
```

`0x415298` disassembles (r2, `pd 40 @ 0x415298`) to:

```
0x00415298  cbz w1, 0x4152a4
0x0041529c  sub w1, w1, 1
0x004152a0  b 0x42ed14
0x004152a4  ldr w8, [x0, 0x20]          ; field+0x20 (an index)
0x004152a8  cmp w8, w2                  ; w2 = argument (label index)
0x004152ac  b.ne 0x4152c0
0x004152b0  ldr s0, [x0, 0x24]          ; s0 = field+0x24
0x004152b4  ldr s1, [x0, 0x3c]          ; s1 = field+0x3c
0x004152b8  fmul s0, s0, s1             ; s0 = field24 * field3c
0x004152bc  ret
0x004152c0  ldr w8, [x0, 0x38]          ; field+0x38 (a second index)
0x004152c4  cmp w8, w2
0x004152c8  b.ne 0x4152e0
0x004152cc  ldp s0, s1, [x0, 0x3c]      ; s0 = field3c, s1 = field+0x40
0x004152d0  fadd s0, s0, s1             ; s0 = field3c + field40
0x004152d4  ldr s1, [x0, 0x24]          ; s1 = field24
0x004152d8  fmul s0, s1, s0             ; s0 = field24 * (field3c + field40)
0x004152dc  ret
0x004152e0  adrp x8, 0xa8000; ldr s0, [x8, 0x318]; ret   ; fallback: fixed constant
```

Field `+0x3c` is proven at construction time (§2) to be a runtime **max-over-array** value
("`first_frame_max`"), and field `+0x40` is copied verbatim from another object's `+0x80` (a strong
candidate for SPEC's "f10", though I did not independently re-derive f10's value here — it is already
pinned in SPEC.md §18). Field `+0x24` is the multiplier applied in **both** branches, which is exactly
SPEC's `-acoustic_scale`. This confirms the *formula* and its address (`0x4152b0..0x4152dc`), independent
of and more precise than the task's hint address.

## 2. Where field `+0x24` (`-acoustic_scale`) comes from: the shared base-class constructor

`0x65f968`'s override belongs to a base class. Chasing `research_handwriting::NetworkScoreCache`'s own
RTTI the same way (name string `N20research_handwriting17NetworkScoreCacheE` at `0xb066d`, typeinfo at
`0x662bc8`, vtable base `0x662b48`) gives vtable slot 3 = `0x42ed14` (the *un-overridden* version of the
same method — see §3). This confirms `PreSpaceAwareNetworkScoreCache : public NetworkScoreCache`.

The constructed object is built in two pieces, both writing into the **same** object (single inheritance,
offset 0 — the field offsets `+0x20`/`+0x24` match between base and derived views):

**Allocation site** (`0x4084f4..0x408500`, inside a `research_handwriting::FstDecoder` method — see §3):

```
0x004084e0  ldr w20, [x22, 0x20]
0x004084e4  cmn w20, 1                  ; w20 == -1 ?
0x004084e8  b.eq 0x408550               ; if so, build plain NetworkScoreCache instead (same base ctor)
0x004084ec  ldr s8, [x22, 0x90]         ; <-- s8 = raw acoustic_scale, read from FstDecoder+0x90
0x004084f0  ldr s9, [x22, 0x80]         ; s9 = another float (candidate f10, copied to field+0x40 later)
0x004084f4  mov w0, 0x48
0x004084f8  bl __lcxx_override          ; operator new(0x48)   -- s8/s9 survive (callee-saved v8/v9)
0x004084fc  bl 0x42b200                 ; construct
```

**`0x42b200`** is a small trampoline (tail call, not a real function):

```
0x0042b200  fmov s0, s8                 ; NB: s8 is callee-saved, so this "receives" 0x4084ec's value
0x0042b204  mov x1, x23
0x0042b208  mov x2, x26
0x0042b20c  mov x29, x0
0x0042b210  b 0x4153b0                  ; tail call into the real (shared) base constructor
```

**`0x4153b0`** (`research_handwriting::NetworkScoreCache::NetworkScoreCache`, confirmed by vtable-install):

```
0x004153b0  sub sp, sp, 0x50
0x004153b4  str d8, [sp, 0x10]
0x004153b8  stp x30, x23, [sp, 0x20]
0x004153bc  stp x22, x21, [sp, 0x30]
0x004153c0  stp x20, x19, [sp, 0x40]
0x004153c4  adrp x8, 0x681000
0x004153c8  mov x19, x1
0x004153cc  mov x21, x0                 ; x21 = this
0x004153d0  ldr x8, [x8, 0xc50]         ; GOT slot -> RELATIVE reloc addend 0x662b48 (vtable base)
0x004153d4  mov x20, x2
0x004153d8  fmov s8, s0                 ; s8 (callee-local) = incoming acoustic_scale argument
0x004153dc  add x8, x8, 0x10
0x004153e0  str x8, [x0]                ; this->vptr = base vtable + 16
...
0x0041547c  fneg s0, s8                 ; s0 = -acoustic_scale
0x00415480  sub w8, w0, 1
0x00415484  mov x0, x19
0x00415488  mov w1, 2
0x0041548c  str w8, [x21, 0x20]         ; field+0x20 = (some label count - 1)
0x00415490  str s0, [x21, 0x24]         ; field+0x24 = -acoustic_scale   <-- THE STORE
```

This is a direct, unambiguous proof that field `+0x24` is `-acoustic_scale`, and that `acoustic_scale`
itself is the float value read at `0x4084ec` from `FstDecoder_instance + 0x90`.

**This is the only construction path in the whole binary.** I searched the full `.text` disassembly (`objdump`,
~1.2M lines) for every direct call to `0x42b200` and every direct branch to `0x4153b0`; there are exactly
three occurrences: `bl 0x42b200` at `0x4084fc` and `0x408558` (two branches of the same `if` inside the one
`FstDecoder` method), and the trampoline's own `b 0x4153b0` at `0x42b210`. No other code in the binary
constructs a `NetworkScoreCache` or `PreSpaceAwareNetworkScoreCache`. (I did not check for virtual-dispatch
construction via a factory pointer table, which a plain `bl`/`b` text search would miss — see "What I could
not determine".)

## 3. `x22` is `research_handwriting::FstDecoder`, and the read site's own method is itself virtual

RTTI for `FstDecoder` (name `N20research_handwriting10FstDecoderE` at `0xad1fc`) resolves to typeinfo
`0x65e9b0`, vtable base `0x65e968`. Vtable slot at `0x65e988` (index 2 of the real methods, after two dtor
slots and one more) = `0x408220` — and the method containing our read site starts there (`0x408220..0x408244`
is a short dispatch/prologue that falls into the `sub sp, sp, #0x430` body I analyzed at `0x408244`, which
contains `0x4084ec`). This confirms `x22` in §2 is a `FstDecoder* this`, and the acoustic_scale read is
inside one of `FstDecoder`'s own virtual methods (plausibly a "build score caches for this utterance" step,
based on the surrounding code building both an ordinary and a space-aware score cache).

## 4. `FstDecoder`'s constructor zero-initializes `+0x90` and does not set it explicitly

`FstDecoder`'s constructor was located the same way (vtable base `0x65e968`, `vtable+16 = 0x65e978`,
found via `adrp x8, 0x65e000 / add x8, x8, 0x978` at `0x40b80c`):

```
0x0040b7e8  sub sp, sp, 0x30
0x0040b7ec  stp x30, x19, [sp, 0x20]
0x0040b7f0  mov w0, 0x120
0x0040b7f4  bl __lcxx_override          ; operator new(0x120)  -- object is 0x120 bytes
0x0040b7f8  mov x19, x0
0x0040b7fc  add x0, x0, 0x20
0x0040b800  mov w1, wzr
0x0040b804  mov w2, 0x100
0x0040b808  bl memset                    ; zero [this+0x20, this+0x120)  <-- covers +0x90
0x0040b80c  adrp x8, 0x65e000
0x0040b810  add x8, x8, 0x978            ; vtable+16
0x0040b814  add x0, sp, 8
0x0040b818  stp xzr, xzr, [x19, 0x10]
0x0040b81c  stp x8, xzr, [x19]           ; this->vptr = vtable+16 ; this+8 = 0
0x0040b820  stp xzr, xzr, [sp, 0x10]
0x0040b824  stp xzr, xzr, [x19, 0x28]
0x0040b828  stp xzr, xzr, [x19, 0x38]
0x0040b82c  str xzr, [sp, 8]
0x0040b830  bl 0x4b4628                  ; a small helper (checked: only touches this+0/0x8/0x10, an mmap
                                          ; buffer reset — not offset 0x90)
0x0040b834  add x0, x19, 0x48
0x0040b838  mov x1, xzr
0x0040b83c  bl 0x4a8980                  ; constructs a sub-object at this+0x48 (not inspected further)
0x0040b840  movi v0.2d, 0
0x0040b844  movi v2.2d, 0xffffffffffffffff
...
0x0040b854  str q1, [x19, 0xd0]
0x0040b858  stur q0, [x19, 0xb8]
0x0040b85c  stur q0, [x19, 0xa8]
0x0040b860  str d2, [x19, 0xc8]
0x0040b864  stp q0, q0, [x19, 0x100]
0x0040b868  ldp x30, x19, [sp, 0x20]
0x0040b86c  b 0x1b2fbc                   ; = "add sp, sp, 0x30; ret" -- ordinary tail-shared epilogue,
                                          ; i.e. the constructor genuinely ends here.
```

Offset `+0x90` falls inside the `memset`-zeroed range `[0x20, 0x120)` and is **never explicitly written** in
this constructor. Every field the constructor *does* set explicitly afterward (`0xa8`, `0xb8`, `0xc8`, `0xd0`,
`0x100..0x120`) is outside `0x90`. This means one of two things: (a) `acoustic_scale` really is populated
later by a separate `Init(FstDecoderConfig)`/configure-style method that I was not able to locate (see
below), or (b) it is read as `0.0` in some code path I have not found (which would be functionally
implausible — it would zero every acoustic score — so I do not believe this, but I cannot rule it out from
static evidence alone).

## What I could not determine

- **The numeric value of `acoustic_scale` for en-US.** I exhaustively searched (`objdump` output, ~1.2M
  disassembled lines) for any instruction anywhere in `.text` that stores a float/double or raw 4/8-byte
  value to `[reg, #0x90]` that could plausibly be `FstDecoder::acoustic_scale_`. I found only 4 float
  stores to a `+0x90` offset anywhere in the whole binary (`0x41f6a4`, `0x487814`, `0x488548`, `0x491970`);
  all four are struct-copy/assignment patterns (they immediately follow a `ldr` from another object's
  `+0x90`, i.e. they propagate an *existing* value, they don't originate one) and none are in the address
  range where `FstDecoder`'s own methods live (`~0x406000-0x40c000`, based on its vtable entries). I also
  checked all `str [w|x]N, [xM, #0x90]` (raw/integer-width) stores in that address range (3 candidates:
  `0x3f2d88`, `0x406fdc`, `0x409eb8`) and confirmed each operates on an unrelated object (different base
  register, unrelated surrounding code — e.g. `0x406fdc` writes a literal `3` into a flags-like object
  returned by an unrelated helper, not into the `FstDecoder` instance). I therefore could not find the
  write site. Plausible explanations I could not rule out: (a) it's set via a register-computed/indexed
  store inside a generic per-field-copy loop (which would not show a literal `#0x90` immediate in the
  disassembly at all — this is how I would try to find it with more time, by tracing forward from the
  `FstDecoderConfig`→runtime-struct transfer function at `0x4ea45c`, which I confirmed copies fields up to
  `~0xf4` but did not have time to fully map field-by-field against `FstDecoder`'s own layout, since the two
  structs do **not** share offsets 1:1 — SPEC's own `+0x44 -> +0x8c` shift for `f7` already establishes the
  proto struct and the "decoder options" struct disagree on offsets, and `FstDecoder` itself is a third,
  even-further-removed layout); or (b) it's set by a call I did not follow (`0x4b4628` and `0x4a8980`,
  called from the constructor, were only partially inspected — `0x4b4628` was ruled out, `0x4a8980`
  (constructing a sub-object at `this+0x48`) was not disassembled at all, and could conceivably reach back
  up to offset `0x90` via a pointer/reference, though `0x90 - 0x48 = 0x48` would be an unusually large
  sub-object for that gap).

- **Whether `acoustic_scale` comes from the recospec proto at all.** I found a compiled-in class
  `speech_decoder::AcousticSearchParams` (RTTI name `N14speech_decoder20AcousticSearchParamsE` at
  `0xc5d4f`, proto message name string `speech_decoder.AcousticSearchParams` at `0x670470`) whose name is
  thematically an exact match for "acoustic scale," and I fully resolved its RTTI/vtable/default
  constructor (`0x4ac1cc`, vtable base `0x6703f0`). However this constructor only initializes a ~0x28-byte
  object (vptr, arena pointer, a zeroed 16-byte block, and one bool-looking byte at `+0x24` set to `1`) —
  far too small to be the backing store read at `FstDecoder+0x90`, and it never writes any float. More
  importantly, `speech_decoder.AcousticSearchParams` **does not appear anywhere** in
  `evidence/proto_message_names.txt`, which SPEC.md and prior work establish as the complete, already-fully-enumerated
  74-message recospec schema. That is reasonably strong (though not conclusive) evidence that this class is
  unrelated dead/shared code statically linked in from a common Google speech-decoder library, and is **not**
  the source of the value read at `FstDecoder+0x90` for the digital-ink recospec pipeline. I could not find
  any recospec field (in `proto/recospec.proto`'s `FstDecoderConfig`/`FstSearchParams`, or in
  `evidence/en_us.recospec.raw.txt`) that is currently unmapped and float-typed in a way that would plausibly
  be `acoustic_scale` — the two unassigned-meaning floats in that submessage (`4.6.7`, `4.6.10`) are already
  pinned to other quantities by SPEC.md §17/§18, and I did not independently re-derive those to check for a
  conflict.

- **The default when the proto field is absent.** Since I could not find the write site, I could not trace
  its default-initialization code path the way SPEC.md §18 did for other fields. The only concrete fact I
  have is that the raw allocated memory at `FstDecoder+0x90` is zero immediately after construction (`memset`,
  §4) and is read verbatim, unconditionally, with no fallback/default logic visible at the read site itself
  (`0x4084e0..0x4084ec` only branches on a *different* field, `+0x20`, to decide which class to build — not
  on `+0x90`). If nothing later overwrites `+0x90`, the value would be `0.0`, which cannot be right
  functionally (it would zero all acoustic scores) — so I believe something else *does* set it, but I
  could not find where.

- I did not attempt to determine whether `FstDecoder`'s virtual `0x408220`/`0x408244` method (which reads
  `+0x90`) is itself reached via a computed/jump-table dispatch — a plain `bl`/`b` text search across all
  1.2M disassembled lines found **zero** direct callers of `0x408244` and **zero** `R_AARCH64_RELATIVE`
  relocations with that address as an addend, meaning it must be reached through a jump table or a virtual
  call I did not chase back further (I confirmed it *is* `FstDecoder` vtable slot `0x65e988`, so virtual
  dispatch through a `Decoder`-family interface pointer is the likely mechanism; I did not chase who holds
  that interface pointer).

## Sanity check on addressing (why I trust this build/offset scheme despite the SPEC.md hint miss)

Before relying on this analysis I verified the file-offset == vaddr assumption and my instruction-boundary
reading against a SPEC.md-cited address that had *not* yet been directly re-verified in this session:
SPEC.md §18 says "options transfer `0x4ea5dc`" for the `4.6.1.7` insertion-penalty field. Disassembling at
exactly `0x4ea5dc` gives:

```
0x004ea5d8  ldrb w10, [x21, 0x2b]
0x004ea5dc  strb w10, [x20, 0x2b]
```

— a byte-field copy inside a larger presence-bitfield-gated struct-copy function (`ldr w9,[x21,0x14]` /
`tst`/`tbnz` chain immediately above, copying dozens of fields from `x21` into `x20` at matching offsets,
gated by has-bits in `w9`). This is exactly the shape SPEC.md describes ("options transfer"), confirming
the file-offset/vaddr scheme and instruction alignment are correct for this exact `.so`. This makes the
`0x4f1182` miss (§ "Summary of findings" above) a genuine, isolated discrepancy rather than a sign that my
whole approach is mis-addressed.

## Answers to the three numbered questions

1. **Value / source**: **Undetermined.** `acoustic_scale` is a runtime float field of a `FstDecoder`
   instance (`this+0x90`), not a compile-time constant embedded in the scoring code itself. I could not
   determine whether it is populated from a recospec proto field (and if so, which one) or from a fixed
   default, and could not determine its numeric value for en-US. See "What I could not determine" for the
   specific dead ends and why (no direct/computable write site found for `FstDecoder+0x90` anywhere in
   `.text`).

2. **Default when absent**: **Undetermined**, for the same reason — I could not locate the write site, so
   I could not trace its default-initialization path (the technique SPEC.md §18 used for other fields).
   The only hard fact is that the raw memory is zero-initialized at construction and not touched again
   within the constructor itself (§4); I do not believe `0.0` is the operative default (it would break
   scoring) but I have no static evidence for what *is*.

3. **Does it scale the net log-posterior before LM combination?** **Yes — proven, not inferred.**
   `research_handwriting::NetworkScoreCache`'s base (non-overridden) per-frame/per-label score accessor,
   at `0x42ed14`, does:
   ```
   0x0042ed6c  sxtw x8, w21                 ; frame index
   0x0042ed70  ldr x9, [x20, 0x18]          ; row stride
   0x0042ed74  ldr s0, [x20, 0x24]          ; s0 = field+0x24 = -acoustic_scale  (same field as §2)
   0x0042ed78  mul x8, x9, x8               ; x8 = frame * stride
   0x0042ed7c  ldr x9, [x20, 8]             ; x9 = base pointer of the net-output/logit array
   0x0042ed80  add x8, x9, x8, lsl 2
   0x0042ed84  ldr s1, [x8, w19, uxtw 2]    ; s1 = net_output[frame*stride + label]  (raw net posterior)
   0x0042ed88  fmul s8, s0, s1              ; result = -acoustic_scale * net_posterior
   ```
   This is the same `field+0x24` established in §2 to be `-acoustic_scale`, and it is applied directly to
   the raw value pulled out of the neural network's output array — i.e. it is the fundamental
   acoustic-score conversion used for every ordinary (non-synthetic-space) frame/label pair throughout
   decoding, not something confined to the pre-space penalty. **If `acoustic_scale != 1.0`, every acoustic
   score the decoder produces is off by that factor relative to the (separately-scaled) LM/char-class costs
   from SPEC.md §17.** This confirms SPEC.md §19's concern is well-founded and precisely locates the
   mechanism, even though the concrete value remains unresolved.

## Confidence

- **High confidence, address-anchored**: the formula and its location (`0x4152b0..0x4152dc`), the fact that
  `field+0x24` is `-acoustic_scale` stored by the shared `NetworkScoreCache` constructor (`0x41547c/0x415490`),
  that this constructor is reached through exactly one code path in the whole binary (§2), that the raw
  value originates from `FstDecoder_instance+0x90` (`0x4084ec`), and the Q3 answer (`0x42ed74..0x42ed88`).
- **Low confidence / explicitly unresolved**: the numeric value of `acoustic_scale`, whether it is
  proto-derived, and its default. These require either dynamic instrumentation (hooking the `0x4153b0`
  constructor or the `0x4084ec` load and running the real Android app) or substantially more static
  data-flow tracing than fit in this session's budget — specifically, tracing `bl 0x4a8980` (called from
  `FstDecoder`'s constructor against `this+0x48`) and locating whatever `FstDecoder::Init`/configure method
  consumes a parsed `FstDecoderConfig`, which I did not locate.
