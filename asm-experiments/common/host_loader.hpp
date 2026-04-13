// ═══════════════════════════════════════════════════════════════════════════════
// host_loader.hpp — Reusable HSACO Loader for Assembly Experiments
// host_loader.hpp — Återanvändbar HSACO-laddare för assembler-experiment
//
// PURPOSE / SYFTE:
//   This header provides a minimal, self-contained utility for loading
//   standalone assembly kernels (.co / .hsaco files) and launching them
//   via the HIP Module API. Every assembly experiment includes this header
//   rather than reimplementing the loading boilerplate.
//
//   Denna header tillhandahåller ett minimalt, fristående verktyg för att
//   ladda fristående assemblerkärnor (.co / .hsaco-filer) och starta dem
//   via HIP Module API. Varje assembler-experiment inkluderar denna header
//   istället för att återimplementera laddningskoden.
//
// USAGE / ANVÄNDNING:
//   #include "../common/host_loader.hpp"
//
//   int main() {
//       AsmKernel kernel("kernel.co", "my_kernel");
//       kernel.launch(grid, block, args, args_size);
//   }
//
// COMPILATION / KOMPILERING:
//   This is plain C++ — compile with amdclang++, NOT hipcc.
//   There is no __global__ device code here; we only link
//   against libamdhip64.so for the runtime API.
//
//   Detta är vanlig C++ — kompilera med amdclang++, INTE hipcc.
//   Det finns ingen __global__-enhetskod här; vi länkar bara
//   mot libamdhip64.so för körtids-API:t.
//
//   amdclang++ -std=c++17 -O2 -I/opt/rocm/include \
//       -L/opt/rocm/lib -lamdhip64 -o runner host.cpp
//
// THE HIP MODULE API — HIP MODULE-API:t:
//   Unlike the <<<>>> launch syntax used with hipcc-compiled kernels,
//   externally assembled kernels must be loaded at runtime using:
//
//   Till skillnad från <<<>>>-startsyntaxen som används med hipcc-kompilerade
//   kärnor, måste externt assemblerade kärnor laddas vid körtid med:
//
//   1. hipModuleLoad(&module, "kernel.co")
//      Loads the HSACO file into GPU memory. The runtime reads the ELF,
//      finds the kernel descriptor, and prepares it for dispatch.
//      Laddar HSACO-filen till GPU-minne.
//
//   2. hipModuleGetFunction(&function, module, "kernel_name")
//      Retrieves a handle to a specific kernel within the module.
//      The name must match the .globl symbol in your .s source.
//      Hämtar ett handtag till en specifik kärna inom modulen.
//      Namnet måste matcha .globl-symbolen i din .s-källa.
//
//   3. hipModuleLaunchKernel(function, gridX, gridY, gridZ,
//                            blockX, blockY, blockZ,
//                            sharedMem, stream, args, nullptr)
//      Dispatches the kernel to the GPU. The args array contains
//      pointers to each kernel argument, matching the order and types
//      declared in your .amdgpu_metadata section.
//      Skickar kärnan till GPU:n. args-arrayen innehåller pekare
//      till varje kärnargument.
//
// ═══════════════════════════════════════════════════════════════════════════════
#pragma once

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>

// ─── Error Checking Macro ────────────────────────────────────────────────────
// ─── Felkontrollmakro ───────────────────────────────────────────────────────
//
// HIP_CHECK wraps every HIP API call. If the call returns an error,
// it prints the error name, file, and line, then aborts.
//
// HIP_CHECK omsluter varje HIP API-anrop. Om anropet returnerar ett fel,
// skriver det ut felnamnet, filen och raden, sedan avbryter det.
//
// WHY abort() AND NOT throw?
// VARFÖR abort() OCH INTE throw?
//   GPU errors are almost always unrecoverable. A failed hipModuleLoad
//   means the code object is invalid. A failed kernel launch means the
//   kernel descriptor is wrong. There is nothing to "catch" — you must
//   fix your assembly source and rebuild.
//   GPU-fel är nästan alltid oåterställbara. Det finns inget att "fånga".
#define HIP_CHECK(call)                                                       \
    do {                                                                      \
        hipError_t err = (call);                                              \
        if (err != hipSuccess) {                                              \
            fprintf(stderr,                                                   \
                "HIP Error / HIP-fel: %s (%d)\n"                              \
                "  at / vid: %s:%d\n"                                         \
                "  call / anrop: %s\n",                                       \
                hipGetErrorString(err), err,                                  \
                __FILE__, __LINE__, #call);                                   \
            abort();                                                          \
        }                                                                     \
    } while (0)


// ─── AsmKernel Class ─────────────────────────────────────────────────────────
// ─── AsmKernel-klass ─────────────────────────────────────────────────────────
//
// Encapsulates the lifetime of a loaded assembly kernel:
// Kapslar in livstiden för en laddad assemblerkärna:
//   - Loads the .co file on construction
//     Laddar .co-filen vid konstruktion
//   - Provides launch() for dispatching
//     Tillhandahåller launch() för dispatch
//   - Unloads the module on destruction
//     Avlastar modulen vid destruktion
//
class AsmKernel {
public:
    // ── Constructor ──────────────────────────────────────────────────────────
    // ── Konstruktor ──────────────────────────────────────────────────────────
    //
    // co_path:      Path to the assembled code object (.co file)
    //               Sökväg till det assemblerade kodobjektet (.co-fil)
    //
    // kernel_name:  Name of the kernel symbol (must match .globl in .s)
    //               Namn på kärnsymbolen (måste matcha .globl i .s)
    //
    AsmKernel(const char* co_path, const char* kernel_name)
        : module_(nullptr), function_(nullptr)
    {
        printf("Loading / Laddar: %s :: %s\n", co_path, kernel_name);

        // hipModuleLoad reads the HSACO ELF file from disk, parses the
        // kernel descriptor, and uploads the kernel code to GPU memory.
        //
        // hipModuleLoad läser HSACO ELF-filen från disk, tolkar
        // kärnbeskrivaren och laddar upp kärnkoden till GPU-minne.
        //
        // Common failure causes / Vanliga felorsaker:
        //   hipErrorFileNotFound (301)  — wrong path / fel sökväg
        //   hipErrorInvalidImage (200)  — corrupt .co or wrong arch / korrupt .co
        //   hipErrorNoBinaryForGpu(209) — built for wrong gfx target
        HIP_CHECK(hipModuleLoad(&module_, co_path));

        // hipModuleGetFunction retrieves the kernel entry point by name.
        // The name must exactly match the .globl symbol in your .s file.
        //
        // hipModuleGetFunction hämtar kärnans startpunkt efter namn.
        // Namnet måste exakt matcha .globl-symbolen i din .s-fil.
        HIP_CHECK(hipModuleGetFunction(&function_, module_, kernel_name));

        printf("  ✅ Loaded successfully / Laddad framgångsrikt\n");
    }

    // ── Destructor ───────────────────────────────────────────────────────────
    // ── Destruktor ───────────────────────────────────────────────────────────
    ~AsmKernel() {
        if (module_) {
            hipModuleUnload(module_);
        }
    }

    // ── Launch ───────────────────────────────────────────────────────────────
    // ── Starta ───────────────────────────────────────────────────────────────
    //
    // Dispatches the kernel with the given grid and block dimensions.
    // Skickar kärnan med givna grid- och blockdimensioner.
    //
    // grid:        Number of workgroups (blockIdx range)
    //              Antal arbetsgrupper
    //
    // block:       Threads per workgroup (threadIdx range)
    //              Trådar per arbetsgrupp
    //
    // args:        Array of void* pointers, one per kernel argument.
    //              Each pointer points to the argument value.
    //              Array av void*-pekare, en per kärnargument.
    //
    // shared_mem:  Dynamic shared memory (LDS) in bytes (usually 0
    //              since assembly kernels declare LDS statically).
    //              Dynamiskt delat minne (LDS) i byte.
    //
    void launch(dim3 grid, dim3 block, void** args,
                size_t shared_mem = 0, hipStream_t stream = nullptr)
    {
        // hipModuleLaunchKernel is the low-level dispatch API.
        // Unlike <<<>>> which the compiler transforms into
        // hipLaunchKernelGGL, this works with externally loaded modules.
        //
        // hipModuleLaunchKernel är det lågnivå-dispatch-API:t.
        // Till skillnad från <<<>>> som kompilatorn transformerar,
        // fungerar detta med externt laddade moduler.
        HIP_CHECK(hipModuleLaunchKernel(
            function_,
            grid.x,  grid.y,  grid.z,     // Grid dimensions / Grid-dimensioner
            block.x, block.y, block.z,     // Block dimensions / Block-dimensioner
            shared_mem,                     // Dynamic shared memory / Dynamiskt delat minne
            stream,                         // Stream (nullptr = default)
            args,                           // Kernel arguments / Kärnargument
            nullptr                         // Extra (unused) / Extra (oanvänd)
        ));
    }

    // ── Timed Launch ─────────────────────────────────────────────────────────
    // ── Tidtagad start ───────────────────────────────────────────────────────
    //
    // Same as launch() but measures kernel execution time in milliseconds.
    // Samma som launch() men mäter kärnans exekveringstid i millisekunder.
    //
    // Returns the elapsed time. Useful for performance experiments.
    // Returnerar förfluten tid. Användbart för prestandaexperiment.
    //
    float launch_timed(dim3 grid, dim3 block, void** args,
                       size_t shared_mem = 0, hipStream_t stream = nullptr)
    {
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        HIP_CHECK(hipEventRecord(start, stream));
        launch(grid, block, args, shared_mem, stream);
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        return ms;
    }

    // ── Accessors ────────────────────────────────────────────────────────────
    hipFunction_t function() const { return function_; }
    hipModule_t   module()   const { return module_; }

    // Non-copyable (module ownership) / Icke-kopierbar (modulägande)
    AsmKernel(const AsmKernel&) = delete;
    AsmKernel& operator=(const AsmKernel&) = delete;

private:
    hipModule_t   module_;
    hipFunction_t function_;
};


// ─── GPU Info Printer ────────────────────────────────────────────────────────
// ─── GPU-informationsutskrivare ──────────────────────────────────────────────
//
// Call this at the start of every experiment to verify you are running
// on the correct GPU. Prints architecture, CU count, and memory.
//
// Anropa detta i början av varje experiment för att verifiera att du
// kör på rätt GPU.
inline void print_gpu_info() {
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));

    printf("═══════════════════════════════════════════════════════\n");
    printf("  GPU: %s\n", props.name);
    printf("  Architecture / Arkitektur: gfx%d\n", props.gcnArchName);
    printf("  Compute Units / Beräkningsenheter: %d\n",
           props.multiProcessorCount);
    printf("  VRAM: %.1f GB\n",
           props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Max threads per block / Max trådar per block: %d\n",
           props.maxThreadsPerBlock);
    printf("  Wavefront size / Wavefront-storlek: %d\n",
           props.warpSize);
    printf("═══════════════════════════════════════════════════════\n\n");
}
