
# asm-01: Hello Assembly — The Minimal Kernel
# asm-01: Hej Assembler — Den minimala kärnan

**Duration / Varaktighet:** 2–3 hours
**Prerequisites / Förutsättningar:** Experiments 01–05 from the main MFMA track
**New concepts / Nya koncept:** Kernel descriptor, metadata, pre-loaded SGPRs, toolchain

---

## Learning Objectives — Lärandemål

By the end of this experiment you will:

1. **Understand the three-section anatomy** of every standalone assembly kernel:
   code, descriptor, metadata.
   *Förstå tresektion-anatomin hos varje fristående assemblerkärna.*

2. **Know exactly which SGPRs** the hardware pre-loads and why.
   *Veta exakt vilka SGPR:er hårdvaran förladdar och varför.*

3. **Be able to assemble, link, and run** a kernel entirely outside of `hipcc`.
   *Kunna assemblera, länka och köra en kärna helt utanför hipcc.*

4. **Read and interpret** `llvm-objdump` and `llvm-readelf` output.
   *Läsa och tolka llvm-objdump- och llvm-readelf-utdata.*

---

## Theory — Teori

### Why Start Here? — Varför börja här?

When you compile a HIP kernel with `hipcc`, the compiler generates:
- Machine instructions (the actual computation)
- A 64-byte kernel descriptor (hardware configuration)
- MessagePack metadata (argument layout for the runtime)

*När du kompilerar en HIP-kärna med hipcc genererar kompilatorn:
maskininstruktioner, en 64-byte kärnbeskrivare och MessagePack-metadata.*

You have never seen these directly because the compiler handles everything.
In assembly, **you must write all three by hand**. Getting any of them wrong
causes silent failures or runtime crashes.

*I assembler måste du skriva alla tre för hand. Fel i någon av dem orsakar
tysta misslyckanden eller körtidskrascher.*

### The Kernel Descriptor — Kärnbeskrivaren

The kernel descriptor is a 64-byte structure placed in `.rodata`.
The HSA runtime reads it **before launching the kernel** to configure:

| Field | Purpose | Consequence of Error |
|-------|---------|---------------------|
| `next_free_vgpr` | How many VGPRs | Too low → corruption. Too high → low occupancy. |
| `next_free_sgpr` | How many SGPRs | Same as above. |
| `kernarg_segment_ptr` | Enable kernarg pointer in s[0:1] | 0 → no access to arguments! |
| `system_sgpr_workgroup_id_x` | Enable blockIdx.x | 0 → all workgroups think they are group 0. |
| `group_segment_fixed_size` | LDS allocation | Mismatch → LDS corruption or launch failure. |
| `wavefront_size32` | Wave32 vs Wave64 | gfx942 doesn't support Wave32 — must be 0. |

### Pre-loaded SGPR Assignment Order — Tilldelningsordning för förladdade SGPR:er

SGPRs are assigned **in the order the flags appear** in the kernel descriptor:

```
1. user_sgpr_private_segment_buffer  (if enabled) → s[0:3]  (4 SGPRs)
2. user_sgpr_dispatch_ptr            (if enabled) → s[N:N+1] (2 SGPRs)
3. user_sgpr_queue_ptr               (if enabled) → s[N:N+1] (2 SGPRs)
4. user_sgpr_kernarg_segment_ptr     (if enabled) → s[N:N+1] (2 SGPRs)
5. user_sgpr_dispatch_id             (if enabled) → s[N:N+1] (2 SGPRs)
   ... then system SGPRs:
6. system_sgpr_workgroup_id_x        (if enabled) → s[N]     (1 SGPR)
7. system_sgpr_workgroup_id_y        (if enabled) → s[N]     (1 SGPR)
8. system_sgpr_workgroup_id_z        (if enabled) → s[N]     (1 SGPR)
```

In our kernel, only `kernarg_segment_ptr` and `workgroup_id_x` are enabled:
- `s[0:1]` = kernarg pointer (64-bit)
- `s2` = workgroup_id_x (32-bit)

*I vår kärna är bara kernarg_segment_ptr och workgroup_id_x aktiverade.*

### The Metadata — Metadata

The `.amdgpu_metadata` section is YAML describing kernel arguments.
The HIP runtime reads this when you call `hipModuleLoad()` to know:
- How many bytes the kernarg buffer needs
- What type each argument is (pointer vs value)
- Byte offsets for argument packing

Without correct metadata, `hipModuleLaunchKernel` will pass garbage to your kernel.

---

## Exercises — Övningar

### Exercise 1: Build and Run — Bygg och kör

```bash
cd asm-01-hello-asm
make asm-01      # from parent directory, or manually:
# /opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx942 -c -o hello_asm.o hello_asm.s
# /opt/rocm/llvm/bin/ld.lld -shared -o hello_asm.co hello_asm.o
# hipcc -std=c++17 -O2 -o runner host.cpp
./runner
```

Expected output: all 64 lanes write 42. ✅

### Exercise 2: Inspect the Code Object — Inspektera kodobjektet

```bash
/opt/rocm/llvm/bin/llvm-objdump -d hello_asm.co
```

Questions / Frågor:
1. How many instructions does the kernel have? *Hur många instruktioner har kärnan?*
2. Can you identify `s_load_dwordx2`, `global_store_dword`, `s_endpgm`?
3. What happens if you change `.amdhsa_next_free_vgpr 3` to `1`?

### Exercise 3: Break It on Purpose — Gör sönder det med avsikt

Try each modification independently and observe the result:
*Prova varje ändring oberoende och observera resultatet:*

1. Remove `s_endpgm` — what happens? *(GPU hang?)*
2. Set `.amdhsa_user_sgpr_kernarg_segment_ptr 0` — what error?
3. Change `.kernarg_segment_size: 8` to `4` in metadata
4. Remove `.p2align 8` before the kernel symbol

### Exercise 4: From Memory — Ur minnet

Close the source file. Write down:
1. The three sections of an assembly kernel file
2. What `s[0:1]` contains when the kernel starts
3. Why `.p2align 8` is required

Then reimplement `hello_asm.s` from memory. Compare your version.
*Stäng källfilen. Skriv ned. Återimplementera sedan ur minnet.*

---

## Key Takeaways — Viktiga insikter

- Every assembly kernel needs THREE sections: `.text`, `.rodata` (descriptor), `.amdgpu_metadata`
  *Varje assemblerkärna behöver TRE sektioner*
- The kernel descriptor configures hardware BEFORE your first instruction runs
  *Kärnbeskrivaren konfigurerar hårdvaran INNAN din första instruktion körs*
- SGPR assignment order depends on which flags are enabled in the descriptor
  *SGPR-tilldelningsordning beror på vilka flaggor som är aktiverade*
- `s_endpgm` is non-negotiable — forgetting it causes a GPU hang
  *s_endpgm är icke-förhandlingsbart*

---

## Next — Nästa

→ **asm-02: Vector Addition** adds global memory loads, multi-argument kernels,
  bounds checking, and thread ID computation.
  *asm-02 lägger till globala minnesladdningar och tråd-ID-beräkning.*
