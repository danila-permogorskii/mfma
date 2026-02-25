// ═══════════════════════════════════════════════════════════════════════════════
// hello_asm.s — The Simplest Possible AMDGPU Assembly Kernel
// hello_asm.s — Den enklaste möjliga AMDGPU-assemblerkärnan
//
// TARGET / MÅL: gfx942 (AMD Instinct MI300X, CDNA3)
//
// PURPOSE / SYFTE:
//   This kernel does absolutely nothing useful. It receives a pointer to a
//   buffer, writes the value 42 to the first element, and terminates.
//   The entire point is to learn the BOILERPLATE — the kernel descriptor,
//   the metadata, and the toolchain — not the computation.
//
//   Denna kärna gör absolut ingenting användbart. Den tar emot en pekare till
//   en buffert, skriver värdet 42 till det första elementet och avslutas.
//   Hela poängen är att lära sig MALLKODEN — kärnbeskrivaren, metadata och
//   verktygskedjan — inte beräkningen.
//
// WHAT YOU WILL LEARN / VAD DU KOMMER ATT LÄRA DIG:
//   1. The .amdgcn_target directive — how to declare your GPU target
//      .amdgcn_target-direktivet — hur du deklarerar ditt GPU-mål
//   2. The .amdhsa_kernel block — the 64-byte kernel descriptor
//      .amdhsa_kernel-blocket — den 64-byte kärnbeskrivaren
//   3. The .amdgpu_metadata section — argument declarations for the runtime
//      .amdgpu_metadata-sektionen — argumentdeklarationer för körtiden
//   4. Pre-loaded SGPRs — how the hardware passes pointers to your kernel
//      Förladdade SGPR:er — hur hårdvaran skickar pekare till din kärna
//   5. s_load_dwordx2 — loading a 64-bit pointer from kernarg memory
//      s_load_dwordx2 — laddning av en 64-bitars pekare från kärnarg-minne
//   6. global_store_dword — writing to device memory
//      global_store_dword — skrivning till enhetsminne
//   7. s_endpgm — the mandatory termination instruction
//      s_endpgm — den obligatoriska avslutningsinstruktionen
//
// BUILD / BYGG:
//   clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx942 -c -o hello_asm.o hello_asm.s
//   ld.lld -shared -o hello_asm.co hello_asm.o
//   amdclang++ -std=c++17 -O2 -I/opt/rocm/include -L/opt/rocm/lib -lamdhip64 -o runner host.cpp
//   ./runner
//
// ═══════════════════════════════════════════════════════════════════════════════


// ─── Section 1: Target Declaration ───────────────────────────────────────────
// ─── Sektion 1: Måldeklaration ───────────────────────────────────────────────
//
// This MUST match the -mcpu flag you pass to the assembler.
// Detta MÅSTE matcha -mcpu-flaggan du skickar till assemblern.
//
// Format: "amdgcn-amd-amdhsa--<chip>"
// The double dash before the chip name is required — it separates the
// environment field (empty) from the processor field.
// Det dubbla bindestrecket före chipnamnet krävs.
//
.amdgcn_target "amdgcn-amd-amdhsa--gfx942"


// ─── Section 2: Kernel Code (.text) ─────────────────────────────────────────
// ─── Sektion 2: Kärnkod (.text) ──────────────────────────────────────────────
//
// The .text section contains the actual machine instructions that the GPU
// will execute. Every instruction here runs on ALL 64 lanes of the wavefront
// simultaneously (unless masked by EXEC).
//
// .text-sektionen innehåller de faktiska maskininstruktionerna som GPU:n
// kommer att exekvera. Varje instruktion här körs på ALLA 64 lanes i
// wavefronten samtidigt (om inte maskerat av EXEC).
//
.text

// .globl makes the symbol visible to the linker. The HIP runtime uses this
// name to find the kernel when you call hipModuleGetFunction().
// .globl gör symbolen synlig för länkaren. HIP-körtiden använder detta
// namn för att hitta kärnan.
.globl hello_asm

// .p2align 8 = align to 2^8 = 256 bytes.
// CDNA3 REQUIRES kernel entry points to be 256-byte aligned.
// The hardware fetches instructions in 256-byte blocks.
// CDNA3 KRÄVER att kärnstartpunkter är 256-byte-justerade.
.p2align 8

// Symbol type declaration — tells the linker this is a function, not data.
// Symboltypdeklaration — talar om för länkaren att detta är en funktion.
.type hello_asm,@function

hello_asm:
    // ═══════════════════════════════════════════════════════════════════════
    // REGISTER MAP — REGISTERKARTA
    //
    // When the kernel starts, the hardware has pre-loaded these registers
    // based on our kernel descriptor settings (Section 3 below):
    //
    // När kärnan startar har hårdvaran förladddat dessa register baserat
    // på våra kärnbeskrivningsinställningar (Sektion 3 nedan):
    //
    //   s[0:1]  = kernarg_segment_ptr  (64-bit pointer to argument buffer)
    //             (64-bitars pekare till argumentbufferten)
    //             This points to a memory region containing our kernel
    //             arguments, laid out according to the metadata in Section 4.
    //
    //   v0      = workitem_id_x        (threadIdx.x — lane ID within workgroup)
    //             (threadIdx.x — lane-ID inom arbetsgrupp)
    //             For a 64-thread workgroup, v0 = 0..63.
    //
    // We will use:
    //   s[2:3]  = loaded output pointer (from kernarg)
    //   v1      = the value to write (42)
    //   v2      = zero offset for the store instruction
    // ═══════════════════════════════════════════════════════════════════════

    // ── Step 1: Load the output pointer from kernel arguments ────────────
    // ── Steg 1: Ladda utdatapekaren från kärnargument ────────────────────
    //
    // s_load_dwordx2 loads two consecutive 32-bit values (= one 64-bit
    // pointer) from the kernarg segment into scalar registers s[2:3].
    //
    // s_load_dwordx2 laddar två på varandra följande 32-bitars värden
    // (= en 64-bitars pekare) från kärnarg-segmentet till skalärregister.
    //
    // Syntax: s_load_dwordx2 dst, base, offset
    //   dst    = s[2:3]   — where to put the loaded pointer
    //   base   = s[0:1]   — kernarg_segment_ptr (pre-loaded by hardware)
    //   offset = 0x0      — byte offset into the kernarg buffer
    //
    // WHY SCALAR? Because the pointer is the SAME for all 64 lanes.
    // Every thread writes to the same buffer — only the offset differs.
    // VARFÖR SKALÄR? Pekaren är SAMMA för alla 64 lanes.
    //
    s_load_dwordx2 s[2:3], s[0:1], 0x0

    // ── Step 2: Wait for the load to complete ────────────────────────────
    // ── Steg 2: Vänta på att laddningen slutförs ─────────────────────────
    //
    // s_load is asynchronous — it issues the request and continues.
    // s_waitcnt lgkmcnt(0) stalls until ALL pending LDS/GDS/Kernarg/Message
    // operations complete. lgkm = LDS, GDS, Kernarg, Message.
    //
    // s_load är asynkron — den utfärdar begäran och fortsätter.
    // s_waitcnt lgkmcnt(0) stoppar tills ALLA väntande LDS/GDS/Kernarg/
    // Message-operationer slutförs.
    //
    // The (0) means "wait until the count of pending operations reaches 0."
    // (0) betyder "vänta tills antalet väntande operationer når 0."
    //
    s_waitcnt lgkmcnt(0)

    // ── Step 3: Prepare the value to write ───────────────────────────────
    // ── Steg 3: Förbered värdet att skriva ───────────────────────────────
    //
    // v_mov_b32 moves a 32-bit immediate value into a vector register.
    // This sets v1 = 42 for ALL 64 lanes simultaneously.
    // "Vector" here means per-lane — but since every lane gets the same
    // immediate, they all hold 42.
    //
    // v_mov_b32 flyttar ett 32-bitars omedelbart värde till ett vektorregister.
    // Detta sätter v1 = 42 för ALLA 64 lanes samtidigt.
    //
    v_mov_b32 v1, 42

    // ── Step 4: Prepare the store offset ─────────────────────────────────
    // ── Steg 4: Förbered lagringsoffset ──────────────────────────────────
    //
    // We want ONLY lane 0 to write (to avoid 64 threads all writing to
    // the same address). We use v0 (workitem_id_x) as a byte offset.
    // Lane 0 has v0=0, so it writes to offset 0. Other lanes write to
    // offset 4, 8, 12, etc. — we only check the first element on the host.
    //
    // Vi vill att BARA lane 0 skriver. Vi använder v0 (workitem_id_x)
    // som en byte-offset multiplicerad med 4.
    //
    // v_lshlrev_b32: left-shift v0 by 2 bits (multiply by 4, since each
    // dword is 4 bytes). Result goes in v2.
    // v_lshlrev_b32: vänsterskifta v0 med 2 bitar (multiplicera med 4).
    //
    v_lshlrev_b32 v2, 2, v0

    // ── Step 5: Write to global memory ───────────────────────────────────
    // ── Steg 5: Skriv till globalt minne ─────────────────────────────────
    //
    // global_store_dword writes a 32-bit value from a VGPR to global memory.
    //
    // global_store_dword skriver ett 32-bitars värde från en VGPR till
    // globalt minne.
    //
    // Syntax: global_store_dword vaddr, vdata, sbase
    //   vaddr  = v2       — per-lane byte offset (lane 0 → 0, lane 1 → 4, ...)
    //   vdata  = v1       — the value to store (42 for all lanes)
    //   sbase  = s[2:3]   — 64-bit base address (our output pointer)
    //
    global_store_dword v2, v1, s[2:3]

    // ── Step 6: Wait for stores and terminate ────────────────────────────
    // ── Steg 6: Vänta på skrivningar och avsluta ─────────────────────────
    //
    // s_waitcnt vmcnt(0) waits for all pending vector memory operations
    // (global loads and stores) to complete.
    // vmcnt = Vector Memory Count.
    //
    // s_waitcnt vmcnt(0) väntar på att alla väntande vektorminnesoperationer
    // slutförs. vmcnt = Vector Memory Count.
    //
    s_waitcnt vmcnt(0)

    // s_endpgm terminates the wavefront. EVERY kernel MUST end with this.
    // If you forget it, the wavefront will execute whatever garbage follows
    // in memory — usually causing a GPU hang.
    //
    // s_endpgm avslutar wavefronten. VARJE kärna MÅSTE sluta med detta.
    // Om du glömmer det kommer wavefronten att exekvera skräp som följer
    // i minnet — vanligtvis orsakar det en GPU-låsning.
    //
    s_endpgm

// Size directive — tells the linker the function's byte length.
// Storleksdirektiv — talar om för länkaren funktionens bytelängd.
.Lhello_asm_end:
    .size hello_asm, .Lhello_asm_end - hello_asm


// ─── Section 3: Kernel Descriptor (.rodata) ──────────────────────────────────
// ─── Sektion 3: Kärnbeskrivare (.rodata) ─────────────────────────────────────
//
// The kernel descriptor is a 64-byte structure that the HSA runtime reads
// BEFORE launching the kernel. It tells the hardware:
//   - How many registers this kernel uses (determines occupancy)
//   - Which SGPRs to pre-load with system values
//   - How much LDS and scratch memory to allocate
//   - The floating-point mode configuration
//
// Kärnbeskrivaren är en 64-byte-struktur som HSA-körtiden läser INNAN
// kärnan startas. Den berättar för hårdvaran:
//   - Hur många register denna kärna använder (bestämmer beläggning)
//   - Vilka SGPR:er som ska förladdas med systemvärden
//   - Hur mycket LDS och scratch-minne som ska allokeras
//   - Konfigurationen av flyttalsläge
//
// The symbol name MUST be <kernel_name>.kd — the assembler creates this
// automatically from the .amdhsa_kernel directive.
// Symbolnamnet MÅSTE vara <kärnnamn>.kd.
//
.rodata
.p2align 6     // Align to 64 bytes (2^6) — required by the ABI
               // Justera till 64 byte — krävs av ABI:t

.amdhsa_kernel hello_asm

    // ── Register counts ──────────────────────────────────────────────────
    // ── Registerräkning ──────────────────────────────────────────────────
    //
    // "next_free" means "the first register number NOT used by this kernel."
    // If you use v0, v1, v2 → next_free_vgpr = 3.
    // If you use s0, s1, s2, s3 → next_free_sgpr = 4.
    //
    // "next_free" betyder "det första registernumret som INTE används."
    //
    // WARNING: If you set these too low, your kernel will produce garbage
    // results or crash. If you set them too high, occupancy decreases
    // (fewer wavefronts can run simultaneously).
    //
    // VARNING: Sätter du dessa för lågt producerar din kärna skräpresultat.
    // Sätter du dem för högt minskar beläggningen.
    //
    // Our usage: v0 (workitem_id), v1 (value 42), v2 (offset) → 3 VGPRs
    //            s0-s3 (kernarg ptr + loaded output ptr) → 4 SGPRs
    //
    .amdhsa_next_free_vgpr 3
    .amdhsa_accum_offset 4
    .amdhsa_next_free_sgpr 4

    // ── Pre-loaded SGPR configuration ────────────────────────────────────
    // ── Förladdad SGPR-konfiguration ─────────────────────────────────────
    //
    // These flags tell the hardware which system values to load into SGPRs
    // before the first instruction executes. The SGPRs are assigned
    // sequentially: the first enabled flag gets s[0:1] (if 64-bit) or s0,
    // the next enabled flag gets the next available SGPR, and so on.
    //
    // Dessa flaggor berättar för hårdvaran vilka systemvärden som ska
    // laddas in i SGPR:er innan den första instruktionen exekveras.
    //
    .amdhsa_user_sgpr_kernarg_segment_ptr 1    // s[0:1] ← kernarg pointer
                                               // 64-bitars pekare till argument

    // These are OFF for this minimal kernel:
    // Dessa är AV för denna minimala kärna:
    .amdhsa_user_sgpr_dispatch_ptr 0           // We don't need the dispatch packet
    .amdhsa_user_sgpr_queue_ptr 0              // We don't need the queue pointer
    .amdhsa_user_sgpr_kernarg_preload_length 0 // No kernarg preloading
    .amdhsa_user_sgpr_kernarg_preload_offset 0
    .amdhsa_system_sgpr_workgroup_id_x 1       // s2 ← workgroup_id_x (after user SGPRs)
    .amdhsa_system_sgpr_workgroup_id_y 0
    .amdhsa_system_sgpr_workgroup_id_z 0

    // ── VGPR work-item ID ────────────────────────────────────────────────
    // 0 = only workitem_id_x (v0). 1 = x+y. 2 = x+y+z.
    // 0 = bara workitem_id_x (v0).
    .amdhsa_system_vgpr_workitem_id 0

    // ── Memory sizes ─────────────────────────────────────────────────────
    // No LDS, no scratch (private) memory needed.
    // Inget LDS, inget scratch-minne behövs.
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0

    // ── Floating-point configuration ─────────────────────────────────────
    // 3 = preserve input denorms and generate output denorms for both
    // 32-bit and 16/64-bit operations. This is the safe default.
    // 3 = bevara indata-denormer och generera utdata-denormer.
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_ieee_mode 1

    // ── Wavefront configuration ──────────────────────────────────────────
    // 0 = 64-thread wavefronts. gfx942 ALWAYS uses wave64.
    // RDNA uses wave32 by default, but CDNA3 does not support it.
    // 0 = 64-tråds wavefronts. gfx942 använder ALLTID wave64.
    //.amdhsa_wavefront_size32 0

.end_amdhsa_kernel


// ─── Section 4: Metadata (.note) ─────────────────────────────────────────────
// ─── Sektion 4: Metadata (.note) ─────────────────────────────────────────────
//
// The metadata tells the HIP runtime about our kernel's arguments.
// When you call hipModuleLaunchKernel with an args array, the runtime
// uses this metadata to pack the arguments into the kernarg buffer
// that s[0:1] will point to.
//
// Metadata berättar för HIP-körtiden om vår kärnas argument.
// När du anropar hipModuleLaunchKernel med en args-array använder
// körtiden denna metadata för att packa argumenten i kärnarg-bufferten.
//
// FORMAT: YAML, wrapped in .amdgpu_metadata / .end_amdgpu_metadata
// The runtime parses this at hipModuleLoad time.
// Körtiden tolkar detta vid hipModuleLoad-tiden.
//
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: hello_asm
    .symbol: hello_asm.kd
    .kernarg_segment_size: 8          # Total bytes: one 8-byte pointer
                                      # Totalt byte: en 8-byte pekare
    .group_segment_fixed_size: 0      # Must match .amdhsa_group_segment_fixed_size
    .private_segment_fixed_size: 0    # Must match .amdhsa_private_segment_fixed_size
    .kernarg_segment_align: 8         # Alignment of kernarg buffer
    .wavefront_size: 64
    .sgpr_count: 4
    .vgpr_count: 3
    .max_flat_workgroup_size: 64      # Maximum threads per workgroup
    .args:
      - {.name: output, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
.end_amdgpu_metadata
