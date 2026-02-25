# AMDGPU Assembly Experiments — Native Kernel Engineering for MI300X
# AMDGPU Assembler-experiment — Inbyggd kärnprogrammering för MI300X

*Standalone GPU assembly programming — from bare metal to production scheduling.*
*Fristående GPU-assemblerprogrammering — från ren hårdvara till produktionsschemaläggning.*

---

## Purpose — Syfte

This is a **parallel branch** within the MFMA course repository, focused exclusively
on writing complete GPU kernels in AMDGPU assembly language targeting gfx942 (MI300X).

*Detta är en **parallell gren** inom MFMA-kursförvaret, fokuserad uteslutande på
att skriva kompletta GPU-kärnor i AMDGPU-assemblerspråk riktat mot gfx942 (MI300X).*

Unlike the main experiment track (01–05) where you write HIP C++ and the compiler
generates assembly, here **you are the compiler**. You control every register,
every instruction, every wait state. There is no safety net.

*Till skillnad från det huvudsakliga experimentspåret (01–05) där du skriver HIP C++
och kompilatorn genererar assembler, här **är du kompilatorn**. Du kontrollerar varje
register, varje instruktion, varje väntetillstånd. Det finns inget skyddsnät.*

### Why This Matters — Varför detta är viktigt

1. **CK contribution readiness** — Composable Kernel's deepest optimisations use
   inline assembly for MFMA scheduling and memory operations. Reading and writing
   these patterns requires fluency in the ISA.
   *CK:s djupaste optimeringar använder inline-assembler. Att läsa och skriva
   dessa mönster kräver flyt i ISA:n.*

2. **Compiler verification** — You cannot judge whether `amdclang++ -O3` is doing
   a good job unless you know what optimal assembly looks like.
   *Du kan inte bedöma om kompilatorn gör ett bra jobb om du inte vet hur optimal
   assembler ser ut.*

3. **Performance ceiling** — The compiler cannot always produce the best code.
   Register allocation heuristics, instruction scheduling constraints, and ABI
   compliance force conservative choices. Hand-written assembly eliminates these.
   *Kompilatorn kan inte alltid producera den bästa koden. Handskriven assembler
   eliminerar dessa begränsningar.*

---

## Prerequisites — Förutsättningar

### Knowledge Requirements — Kunskapskrav

You **must** have completed experiments 01–05 from the main track before starting
these assembly experiments. Specifically, you need:

*Du **måste** ha genomfört experiment 01–05 från huvudspåret innan du börjar.
Specifikt behöver du:*

- Solid understanding of the wavefront execution model (64 threads in lockstep)
  *Gedigen förståelse för wavefront-exekveringsmodellen*
- MFMA instruction semantics: D = A × B + C, VGPR inputs, AGPR accumulators
  *MFMA-instruktionssemantik*
- LDS (Local Data Share) mechanics: bank conflicts, synchronisation barriers
  *LDS-mekanik: bankkonflikter, synkroniseringsbarriärer*
- Comfort reading compiler-generated assembly from `hipcc -S` output
  *Bekvämlighet med att läsa kompilatorgenererad assembler*

### Reference Materials — Referensmaterial

Keep these open at all times during assembly work:

| Document | Location | Sections |
|----------|----------|----------|
| **MI300 ISA Manual** | Project knowledge (PDF) | §4 Kernel State, §5 Scalar ALU, §6 Vector ALU, §7 MFMA, §9 LDS |
| **LLVM AMDGPU Backend Guide** | [llvm.org/docs/AMDGPUUsage.html](https://llvm.org/docs/AMDGPUUsage.html) | Kernel Descriptor, Assembler Directives, Code Object Format |
| **gfx942 Instruction Syntax** | [llvm.org/docs/AMDGPU/AMDGPUAsmGFX940.html](https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX940.html) | Full instruction reference |
| **AMDGPU Operand Syntax** | [llvm.org/docs/AMDGPUOperandSyntax.html](https://llvm.org/docs/AMDGPUOperandSyntax.html) | Register notation, alignment rules |
| **GPUOpen Assembly Guide** | [gpuopen.com/learn/amdgcn-assembly/](https://gpuopen.com/learn/amdgcn-assembly/) | Standalone kernel workflow (older but conceptually valid) |

---

## Environment Setup — Miljöinställning

### Required Components — Nödvändiga komponenter

The assembly workflow uses different tools than the HIP C++ track. Verify every
component below exists on your MI300X instance **before** starting any experiment.

*Assembler-arbetsflödet använder andra verktyg än HIP C++-spåret. Verifiera att
varje komponent nedan finns på din MI300X-instans **innan** du börjar.*

#### 1. ROCm Installation (≥ 6.0, tested on 7.1.0)

```bash
# Verify ROCm is installed and GPU is visible
# Verifiera att ROCm är installerat och GPU:n är synlig
rocminfo | grep -E "Name:.*gfx"
# Expected / Förväntat: Name: gfx942
```

#### 2. LLVM Assembler (`llvm-mc`)

The LLVM Machine Code tool assembles `.s` source files into ELF object files.
This is the core of the assembly workflow — it replaces `hipcc` for device code.

*LLVM Machine Code-verktyget assemblerar `.s`-källfiler till ELF-objektfiler.
Detta är kärnan i assembler-arbetsflödet — det ersätter `hipcc` för enhetskod.*

```bash
# Check llvm-mc exists and supports gfx942
# Kontrollera att llvm-mc finns och stödjer gfx942
/opt/rocm/llvm/bin/llvm-mc --version
/opt/rocm/llvm/bin/llvm-mc --arch=amdgcn --mcpu=gfx942 --show-encoding /dev/null

# If the above fails, check alternative paths:
# Om ovanstående misslyckas, kontrollera alternativa sökvägar:
which llvm-mc
ls /opt/rocm/llvm/bin/llvm-mc
```

#### 3. Linker (`ld.lld`)

The LLVM linker produces the final HSACO (HSA Code Object) — a shared ELF
that the HIP runtime can load and dispatch.

*LLVM-länkaren producerar det slutliga HSACO — en delad ELF som HIP-körtiden
kan ladda och skicka.*

```bash
# Verify lld is available
# Verifiera att lld finns tillgänglig
/opt/rocm/llvm/bin/ld.lld --version

# Alternative: the system lld may also work
ld.lld --version
```

#### 4. Disassembler (`llvm-objdump`)

For inspecting assembled code objects — essential for verifying your assembly
was encoded correctly.

*För att inspektera assemblerade kodobjekt — väsentligt för att verifiera att
din assembler kodades korrekt.*

```bash
# Verify llvm-objdump supports AMDGPU
# Verifiera att llvm-objdump stödjer AMDGPU
/opt/rocm/llvm/bin/llvm-objdump --version

# Quick test: disassemble an existing HIP binary
# Snabbtest: disassemblera en befintlig HIP-binär
echo '__global__ void k(){}' > /tmp/test.hip
hipcc --offload-arch=gfx942 -O3 -o /tmp/test /tmp/test.hip
/opt/rocm/llvm/bin/llvm-objdump -d /tmp/test --offloading
```

#### 5. HIP Runtime (host-side kernel loading)

The host code that loads `.hsaco` files and launches kernels uses the HIP
Module API: `hipModuleLoad`, `hipModuleGetFunction`, `hipModuleLaunchKernel`.

*Värdkoden som laddar `.hsaco`-filer och startar kärnor använder HIP Module API.*

```bash
# Verify HIP compiler works (for host code compilation)
# Verifiera att HIP-kompilatorn fungerar (för kompilering av värdkod)
hipcc --version

# Verify the HIP runtime library is linkable
# Verifiera att HIP-körtidsbiblioteket kan länkas
ls /opt/rocm/lib/libamdhip64.so
```

#### 6. Code Object Inspection Tools

```bash
# roc-obj-ls — lists kernels inside code objects (may be deprecated, use llvm-objdump)
# llvm-readelf — reads ELF headers and note records (metadata)
/opt/rocm/llvm/bin/llvm-readelf --version

# Verify we can read note records from a code object
# Verifiera att vi kan läsa anteckningsposter från ett kodobjekt
```

#### 7. Profiling Tools (same as main track)

```bash
# rocprofv3 — kernel timing and hardware counters
# rocprofv3 — kärntiming och hårdvaruräknare
which rocprofv3
rocprofv3 --version

# rocprof-compute — roofline analysis
# rocprof-compute — taklinjeanalys
which rocprof-compute
```

---

### Environment Verification Script — Miljöverifieringsskript

Run this script to confirm everything is in place before you begin.

*Kör detta skript för att bekräfta att allt finns på plats innan du börjar.*

```bash
#!/bin/bash
# verify_asm_env.sh — Assembly experiment environment checker
# verify_asm_env.sh — Kontroll av assembler-experimentmiljö

echo "═══════════════════════════════════════════════════════════"
echo "  AMDGPU Assembly Environment Verification"
echo "  Verifiering av AMDGPU-assemblermiljö"
echo "═══════════════════════════════════════════════════════════"

ROCM=/opt/rocm
PASS=0
FAIL=0

check() {
    if eval "$2" > /dev/null 2>&1; then
        echo "  ✅ $1"
        ((PASS++))
    else
        echo "  ❌ $1 — MISSING / SAKNAS"
        ((FAIL++))
    fi
}

echo ""
echo "── GPU Hardware ──"
check "gfx942 detected"       "rocminfo 2>/dev/null | grep -q 'gfx942'"

echo ""
echo "── Assembly Toolchain (Assembler-verktygskedja) ──"
check "llvm-mc (assembler)"    "test -x ${ROCM}/llvm/bin/llvm-mc"
check "ld.lld (linker)"        "test -x ${ROCM}/llvm/bin/ld.lld"
check "llvm-objdump"           "test -x ${ROCM}/llvm/bin/llvm-objdump"
check "llvm-readelf"           "test -x ${ROCM}/llvm/bin/llvm-readelf"
check "clang (for asm)"        "test -x ${ROCM}/llvm/bin/clang"

echo ""
echo "── HIP Host Tools (HIP-värdverktyg) ──"
check "hipcc (host compiler)"  "which hipcc"
check "libamdhip64.so"         "test -f ${ROCM}/lib/libamdhip64.so"

echo ""
echo "── Profiling (Profilering) ──"
check "rocprofv3"              "which rocprofv3"

echo ""
echo "── Code Object Inspection (Kodobjektinspektion) ──"
check "llvm-objdump --offloading" \
    "${ROCM}/llvm/bin/llvm-objdump --help 2>&1 | grep -q offloading"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Results / Resultat: ${PASS} passed, ${FAIL} failed"
echo "═══════════════════════════════════════════════════════════"

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "  ⚠  Some components are missing. Install ROCm ≥ 6.0 with"
    echo "     the full LLVM toolchain to proceed."
    echo "  ⚠  Vissa komponenter saknas. Installera ROCm ≥ 6.0 med"
    echo "     den fullständiga LLVM-verktygskedjan för att fortsätta."
    exit 1
fi
```

Save this as `verify_asm_env.sh` in the assembly experiments root directory.
Run it on your MI300X instance before starting experiment asm-01.

*Spara detta som `verify_asm_env.sh`. Kör det på din MI300X-instans innan
du börjar experiment asm-01.*

---

## Toolchain Pipeline — Verktygskedjans pipeline

Understanding the full build pipeline is essential. This is what replaces
the single `hipcc` command you are accustomed to.

*Att förstå den fullständiga bygg-pipelinen är väsentligt. Detta ersätter det
enda `hipcc`-kommandot du är van vid.*

```
   Assembly Source (.s)          Host Source (.cpp)
         │                              │
         ▼                              │
   ┌──────────────┐                     │
   │   llvm-mc     │                     │
   │  --arch=amdgcn│                     │
   │  --mcpu=gfx942│                     │
   │  -filetype=obj│                     │
   └──────┬───────┘                     │
          │ kernel.o                    │
          ▼                             │
   ┌──────────────┐                     │
   │   ld.lld      │                     │
   │  -shared      │                     │
   │  -o kernel.co │                     │
   └──────┬───────┘                     │
          │ kernel.co (HSACO)           │
          │                             ▼
          │                    ┌──────────────┐
          │                    │  amdclang++   │
          │                    │  + libamdhip64│
          │                    │  host.cpp     │
          │                    └──────┬───────┘
          │                           │ host (executable)
          ▼                           ▼
   ┌──────────────────────────────────────┐
   │           HIP Runtime                 │
   │  hipModuleLoad("kernel.co")          │
   │  hipModuleGetFunction(module, "name")│
   │  hipModuleLaunchKernel(fn, ...)      │
   └──────────────────────────────────────┘
```

### The Three-Step Build — Trestegsbygget

```bash
# Step 1: Assemble — Steg 1: Assemblera
# Converts your .s source to an ELF object file
# Konverterar din .s-källa till en ELF-objektfil
/opt/rocm/llvm/bin/clang -x assembler \
    -target amdgcn-amd-amdhsa \
    -mcpu=gfx942 \
    -c -o kernel.o kernel.s

# Step 2: Link — Steg 2: Länka
# Produces the HSACO (HSA Code Object) shared library
# Producerar HSACO (HSA Code Object) delat bibliotek
/opt/rocm/llvm/bin/ld.lld \
    -shared \
    -o kernel.co kernel.o

# Step 3: Compile host — Steg 3: Kompilera värd
# Plain C++ linking against libamdhip64 — NOT hipcc.
# The host code has NO device code; it just calls the HIP runtime API.
# Vanlig C++ som länkas mot libamdhip64 — INTE hipcc.
# Värdkoden har INGEN enhetskod; den anropar bara HIP-körtids-API:t.
/opt/rocm/llvm/bin/amdclang++ -std=c++17 -O2 \
    -I/opt/rocm/include \
    -L/opt/rocm/lib -lamdhip64 \
    -o runner host.cpp

# Run — Kör
./runner
```

### Alternative: Direct `llvm-mc` Assembly

```bash
# llvm-mc is the lower-level assembler. clang wraps it with additional
# preprocessing (macros, includes). For pure .s files, either works.
# llvm-mc är den lägre assemblern. clang omsluter den med ytterligare
# förbearbetning. För rena .s-filer fungerar båda.
/opt/rocm/llvm/bin/llvm-mc \
    -arch=amdgcn \
    -mcpu=gfx942 \
    -filetype=obj \
    -o kernel.o kernel.s
```

### Inspecting Your Output — Inspektera din utdata

```bash
# Disassemble the code object to verify instructions
# Disassemblera kodobjektet för att verifiera instruktioner
/opt/rocm/llvm/bin/llvm-objdump -d kernel.co

# Read the kernel descriptor and metadata
# Läs kärnbeskrivaren och metadata
/opt/rocm/llvm/bin/llvm-readelf -n kernel.co

# Show all sections and symbols
# Visa alla sektioner och symboler
/opt/rocm/llvm/bin/llvm-readelf -S -s kernel.co
```

---

## Directory Structure — Katalogstruktur

```
asm-experiments/
├── README.md                 ← This file (du läser den nu)
├── verify_asm_env.sh         ← Environment verification script
├── common/
│   └── host_loader.cpp       ← Reusable HSACO loader + launcher
│                               (Återanvändbar HSACO-laddare + startare)
├── Makefile                  ← Top-level build for all asm experiments
│
├── asm-01-hello-asm/
│   ├── README.md             ← Theory: kernel descriptor, ABI, metadata
│   ├── hello_asm.s           ← Minimal gfx942 kernel in pure assembly
│   ├── host.cpp              ← Host code using hipModuleLoad
│   └── Makefile              ← Assemble → link → compile host → run
│
├── asm-02-vector-add/
│   ├── README.md             ← Theory: addressing modes, global_load/store
│   ├── vector_add.s          ← Vector addition with manual register allocation
│   ├── host.cpp              ← Host verification with CPU reference
│   └── Makefile
│
├── asm-03-lds-manual/
│   ├── README.md             ← Theory: ds_read, ds_write, s_waitcnt lgkmcnt
│   ├── lds_reduce.s          ← Manual LDS reduction kernel
│   ├── host.cpp
│   └── Makefile
│
├── asm-04-mfma-bare/
│   ├── README.md             ← Theory: MFMA encoding, AGPR allocation, data layout
│   ├── mfma_bare.s           ← Single v_mfma_f32_16x16x16_f16 with hand-scheduled loads
│   ├── host.cpp
│   └── Makefile
│
├── asm-05-mfma-pipeline/
│   ├── README.md             ← Theory: double buffering, instruction interleaving
│   ├── mfma_pipeline.s       ← Production-style interleaved MFMA + global_load pipeline
│   ├── host.cpp
│   └── Makefile
│
└── asm-06-mfma-vs-compiler/
    ├── README.md             ← Theory: profiling comparison methodology
    ├── mfma_hand.s           ← Hand-optimised MFMA GEMM
    ├── mfma_compiler.cpp     ← Equivalent HIP C++ for compiler comparison
    ├── host.cpp
    └── Makefile
```

### Incremental Design — Inkrementell design

Each experiment builds directly upon the previous one. *Varje experiment bygger
direkt på det föregående.* The progression is:

| Experiment | Builds Upon | New Concept | Du lär dig |
|------------|-------------|-------------|------------|
| **asm-01** | Nothing — ground zero | Kernel descriptor, metadata, ABI format, `s_endpgm` | Hur en kärna överhuvudtaget startar |
| **asm-02** | asm-01 descriptor template | `global_load_dwordx4`, `global_store_dword`, `s_waitcnt`, addressing | Minnesåtkomst och registerallokering |
| **asm-03** | asm-02 memory patterns | `ds_read_b32`, `ds_write_b32`, `s_barrier`, `s_waitcnt lgkmcnt(0)` | Delat minne utan kompilatorhjälp |
| **asm-04** | asm-03 LDS + asm-01 descriptor | `v_mfma_f32_16x16x16_f16`, AGPR initialisation, VGPR→AGPR movement | MFMA-instruktioner utan abstraktion |
| **asm-05** | asm-04 MFMA + asm-03 LDS | Interleaved load/compute, double-buffered LDS, `s_waitcnt vmcnt(N)` tuning | Produktionsmönster för latensföljning |
| **asm-06** | All previous | rocprofv3 comparison, `SQ_INSTS_MFMA` counters, TFLOPS measurement | Bevisbar prestandafördel |

---

## The Assembly Source File Anatomy — Assemblerfilens anatomi

Every standalone AMDGPU kernel in assembly has this structure.
Study this before starting asm-01. *Studera detta innan du börjar asm-01.*

```asm
// ═══════════════════════════════════════════════════════════════
// FILE STRUCTURE OF AN AMDHSA ASSEMBLY KERNEL
// FILSTRUKTUR FÖR EN AMDHSA-ASSEMBLERKÄRNA
// ═══════════════════════════════════════════════════════════════

// ─── Section 1: Target Declaration ───────────────────────────
// Tells the assembler which GPU architecture we target.
// Talar om för assemblern vilken GPU-arkitektur vi riktar mot.
.amdgcn_target "amdgcn-amd-amdhsa--gfx942"

// ─── Section 2: Kernel Code ──────────────────────────────────
// The .text section contains the actual machine instructions.
// .text-sektionen innehåller de faktiska maskininstruktionerna.
.text
.globl my_kernel            // Make the symbol visible to the linker
.p2align 8                  // Align to 256 bytes (required by hardware)
.type my_kernel,@function

my_kernel:
    // Your instructions go here
    // Dina instruktioner placeras här
    s_endpgm                // End program — every kernel MUST end with this
                            // Avsluta program — varje kärna MÅSTE sluta med detta
.Lmy_kernel_end:
    .size my_kernel, .Lmy_kernel_end - my_kernel

// ─── Section 3: Kernel Descriptor ────────────────────────────
// The 64-byte structure that tells the HSA runtime how to
// configure hardware before launching this kernel.
// 64-byte-strukturen som talar om för HSA-körtiden hur
// hårdvaran ska konfigureras innan denna kärna startas.
.rodata                     // Note: newer LLVM uses .amdhsa.kd section
.p2align 6                  // Align to 64 bytes
.amdhsa_kernel my_kernel
    // ── Register allocation ──
    // How many VGPRs and SGPRs does this kernel use?
    // Hur många VGPR:er och SGPR:er använder denna kärna?
    .amdhsa_next_free_vgpr <N>    // Number of VGPRs used (determines occupancy)
    .amdhsa_next_free_sgpr <M>    // Number of SGPRs used

    // ── Pre-loaded SGPRs ──
    // Which system values should the hardware load into SGPRs
    // before the first instruction executes?
    // Vilka systemvärden ska hårdvaran ladda in i SGPR:er
    // innan den första instruktionen exekveras?
    .amdhsa_user_sgpr_kernarg_segment_ptr 1   // Pointer to kernel arguments
    .amdhsa_user_sgpr_dispatch_ptr 0          // Dispatch packet pointer
    .amdhsa_user_sgpr_workgroup_id_x 1        // blockIdx.x in SGPR
    .amdhsa_user_sgpr_workgroup_id_y 0        // blockIdx.y (off if 1D)
    .amdhsa_user_sgpr_workgroup_id_z 0        // blockIdx.z (off if 1D)

    // ── Wavefront configuration ──
    .amdhsa_system_vgpr_workitem_id 0         // 0 = X only, 1 = X+Y, 2 = X+Y+Z

    // ── Memory configuration ──
    .amdhsa_group_segment_fixed_size 0        // LDS size in bytes (0 = no LDS)
    .amdhsa_private_segment_fixed_size 0      // Scratch size per work-item (0 = no scratch)

    // ── Floating-point modes ──
    .amdhsa_float_denorm_mode_32 3            // 3 = flush/preserve denorms
    .amdhsa_float_denorm_mode_16_64 3

    // ── AGPR count (CDNA-specific) ──
    .amdhsa_accum_offset <offset>             // First AGPR = VGPR_count (architecture specific)

    .amdhsa_ieee_mode 1
    .amdhsa_wavefront_size32 0                // 0 = 64-thread wavefronts (always for gfx942)

.end_amdhsa_kernel

// ─── Section 4: Metadata ─────────────────────────────────────
// MessagePack-encoded metadata describing kernel arguments.
// The HIP runtime reads this to set up the dispatch.
// MessagePack-kodad metadata som beskriver kärnargument.
// HIP-körtiden läser detta för att konfigurera dispatch.
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: my_kernel
    .symbol: my_kernel.kd
    .kernarg_segment_size: <size>     // Total size of all arguments in bytes
    .group_segment_fixed_size: 0      // Must match .amdhsa_group_segment_fixed_size
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: <M>
    .vgpr_count: <N>
    .max_flat_workgroup_size: 256     // Maximum threads per workgroup
    .args:
      - {.name: arg0, .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global}
      - {.name: arg1, .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global}
      - {.name: arg2, .size: 4, .offset: 16, .value_kind: by_value}
.end_amdgpu_metadata
```

---

## Register Quick Reference for gfx942 — Registersnabbreferens

| Register Class | Notation | Count | Width | Purpose |
|----------------|----------|-------|-------|---------|
| **SGPR** | `s0`–`s101` | 102 | 32-bit | Scalars: pointers, loop counters, constants. Shared across entire wavefront. *Skalärer: pekare, loopräknare, konstanter. Delas över hela wavefronten.* |
| **VGPR** | `v0`–`v255` | 256 | 32-bit | Vectors: per-lane data. 64 physical values per VGPR (one per lane). *Vektorer: per-lane-data.* |
| **AGPR** | `a0`–`a255` | 256 | 32-bit | Accumulators: MFMA output registers. Cannot be used for general ALU. *Ackumulatorer: MFMA-utdata.* |
| **VCC** | `vcc` | 1 | 64-bit | Vector condition code: result of vector comparisons. *Vektorvillkorskod.* |
| **EXEC** | `exec` | 1 | 64-bit | Execution mask: which lanes are active. *Exekveringsmask: vilka lanes som är aktiva.* |
| **SCC** | `scc` | 1 | 1-bit | Scalar condition code. *Skalär villkorskod.* |
| **M0** | `m0` | 1 | 32-bit | Miscellaneous register: LDS access control. *Diverse register.* |

### gfx942-Specific Alignment Rules — Justeringsregler specifika för gfx942

- **VGPR pairs** must be even-aligned: `v[0:1]` ✅, `v[1:2]` ❌
  *VGPR-par måste vara jämt justerade*
- **AGPR pairs** must be even-aligned: `a[0:1]` ✅, `a[1:2]` ❌
  *AGPR-par måste vara jämt justerade*
- **SGPR sequences** of 4+ must be quad-aligned: `s[0:3]` ✅, `s[1:4]` ❌
  *SGPR-sekvenser av 4+ måste vara quad-justerade*
- **64-bit SGPR pairs** must be even-aligned: `s[0:1]` ✅, `s[1:2]` ❌
  *64-bitars SGPR-par måste vara jämt justerade*

### Pre-loaded SGPRs — Förladdade SGPR:er

When the hardware launches your kernel, certain SGPRs are pre-loaded
based on your kernel descriptor settings:

*När hårdvaran startar din kärna förladdas vissa SGPR:er baserat på dina
kärnbeskrivningsinställningar:*

```
Typical SGPR layout for a kernel with kernarg pointer + workgroup ID:

s[0:1]  ← kernarg_segment_ptr (64-bit pointer to kernel arguments)
           (64-bitars pekare till kärnargument)
s2      ← workgroup_id_x (blockIdx.x)
           (arbetsgruppens ID i X)

The exact assignment depends on which .amdhsa_user_sgpr_* flags you enable.
SGPRs are assigned in the order the flags appear in the kernel descriptor,
starting from s0.

Den exakta tilldelningen beror på vilka .amdhsa_user_sgpr_*-flaggor du aktiverar.
SGPR:er tilldelas i den ordning flaggorna visas i kärnbeskrivaren, från s0.
```

### Pre-loaded VGPRs — Förladdade VGPR:er

```
v0  ← workitem_id_x (threadIdx.x, always loaded)
       (arbetsartikelns ID i X, alltid laddad)

If .amdhsa_system_vgpr_workitem_id >= 1:
v1  ← workitem_id_y (threadIdx.y)

If .amdhsa_system_vgpr_workitem_id >= 2:
v2  ← workitem_id_z (threadIdx.z)
```

---

## Coding Conventions — Kodkonventioner

### Assembly Source Files

- **File extension**: `.s` (not `.asm` — the LLVM toolchain expects `.s`)
  *Filändelse: `.s` (inte `.asm`)*
- **Comments**: `//` for line comments (C++ style, supported by LLVM assembler)
  *Kommentarer: `//` för radkommentarer*
- **Bilingual comments**: English first, Swedish translation in parentheses
  *Tvåspråkiga kommentarer: engelska först, svensk översättning inom parentes*
- **Register naming**: document every register's purpose in a header block
  *Registernamngivning: dokumentera varje registers syfte i ett huvudblock*
- **Section separators**: use `// ═══` for major sections, `// ───` for subsections
  *Sektionsavgränsare*

### Host Source Files

- Same conventions as the main MFMA course (C++17, `HIP_CHECK` macros)
  *Samma konventioner som huvud-MFMA-kursen*
- Compiled with `amdclang++`, NOT `hipcc` — no device code in host files
  *Kompilerade med `amdclang++`, INTE `hipcc` — ingen enhetskod i värdfiler*
- Must use `hipModule*` API (not `<<<>>>` syntax) since kernels are external
  *Måste använda `hipModule*`-API:t eftersom kärnorna är externa*

---

## How to Work Through These Experiments — Hur du arbetar genom experimenten

Follow the **Deliberate Transcription Protocol** from the main course:

*Följ **Avsiktlig transkriptionsprotokollet** från huvudkursen:*

1. **Read** the experiment README thoroughly. Understand the theory.
   *Läs experimentets README grundligt. Förstå teorin.*

2. **Study** the provided `.s` source with the ISA manual open beside you.
   *Studera den medföljande `.s`-källan med ISA-manualen öppen bredvid dig.*

3. **Close** the source. Write down what you remember about the register layout,
   the instruction sequence, and the kernel descriptor.
   *Stäng källan. Skriv ner vad du minns.*

4. **Type** the assembly from memory. Make mistakes. Debug them with `llvm-mc` errors
   and `llvm-objdump` output.
   *Skriv assemblern ur minnet. Gör misstag. Felsök dem.*

5. **Compare** your version against the reference. Document the delta in Obsidian.
   *Jämför din version mot referensen. Dokumentera deltat i Obsidian.*

6. **Measure** with `rocprofv3`. Numbers don't lie.
   *Mät med `rocprofv3`. Siffror ljuger inte.*

---

## References — Referenser

| Resource | URL | Notes |
|----------|-----|-------|
| AMD MI300 ISA | Project knowledge | The authoritative reference. §7 for MFMA. |
| LLVM AMDGPU Usage | [llvm.org/docs/AMDGPUUsage.html](https://llvm.org/docs/AMDGPUUsage.html) | Kernel descriptor format, code object ABI |
| LLVM gfx942 Instructions | [llvm.org/docs/AMDGPU/AMDGPUAsmGFX940.html](https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX940.html) | Every instruction's syntax |
| LLVM Operand Syntax | [llvm.org/docs/AMDGPUOperandSyntax.html](https://llvm.org/docs/AMDGPUOperandSyntax.html) | Register alignment rules |
| GPUOpen Assembly Guide | [gpuopen.com/learn/amdgcn-assembly/](https://gpuopen.com/learn/amdgcn-assembly/) | Conceptual workflow (older arch, still valid concepts) |
| ROCm LLVM Assembler Extra | [github.com/ROCm/LLVM-AMDGPU-Assembler-Extra](https://github.com/ROCm/LLVM-AMDGPU-Assembler-Extra) | Example standalone assembly kernels |
| LLVM Issue #131954 | [github.com/llvm/llvm-project/issues/131954](https://github.com/llvm/llvm-project/issues/131954) | Compiler AGPR spill problems with large MFMA tiles |
