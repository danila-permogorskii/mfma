// ═══════════════════════════════════════════════════════════════════════════════
// host.cpp — Host Program for asm-01: Hello Assembly
// host.cpp — Värdprogram för asm-01: Hej Assembler
//
// This program loads the hand-written assembly kernel from hello_asm.co,
// launches it on the GPU, and verifies the result.
//
// Detta program laddar den handskrivna assemblerkärnan från hello_asm.co,
// startar den på GPU:n och verifierar resultatet.
// ═══════════════════════════════════════════════════════════════════════════════
#include "../common/host_loader.hpp"
#include <cstring>

int main() {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  asm-01: Hello Assembly — The Minimal Kernel\n");
    printf("  asm-01: Hej Assembler — Den minimala kärnan\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    print_gpu_info();

    // ── Allocate device memory ───────────────────────────────────────────
    // ── Allokera enhetsminne ─────────────────────────────────────────────
    const int N = 64;  // One element per lane in the wavefront
    int* d_output = nullptr;
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(int)));
    HIP_CHECK(hipMemset(d_output, 0, N * sizeof(int)));

    // ── Load and launch the assembly kernel ──────────────────────────────
    // ── Ladda och starta assemblerkärnan ──────────────────────────────────
    AsmKernel kernel("hello_asm.co", "hello_asm");

    // Set up kernel arguments:
    // The kernarg buffer will contain just one 8-byte pointer (d_output).
    // Kärnarg-bufferten innehåller bara en 8-byte pekare.
    void* args[] = { &d_output };

    // Launch: 1 workgroup × 64 threads = 1 wavefront
    // Starta: 1 arbetsgrupp × 64 trådar = 1 wavefront
    float ms = kernel.launch_timed(dim3(1), dim3(64), args);
    HIP_CHECK(hipDeviceSynchronize());

    printf("  Kernel executed in %.3f ms\n", ms);
    printf("  Kärnan exekverades på %.3f ms\n\n", ms);

    // ── Copy results back and verify ─────────────────────────────────────
    // ── Kopiera tillbaka resultat och verifiera ──────────────────────────
    int h_output[N];
    HIP_CHECK(hipMemcpy(h_output, d_output, N * sizeof(int), hipMemcpyDeviceToHost));

    printf("  Results (first 8 lanes) / Resultat (första 8 lanes):\n");
    for (int i = 0; i < 8; i++) {
        printf("    lane %d: %d %s\n", i, h_output[i],
               h_output[i] == 42 ? "✅" : "❌");
    }

    // Verify ALL lanes wrote 42
    bool all_correct = true;
    for (int i = 0; i < N; i++) {
        if (h_output[i] != 42) { all_correct = false; break; }
    }

    printf("\n  %s All %d lanes wrote 42\n",
           all_correct ? "✅" : "❌", N);
    printf("  %s Alla %d lanes skrev 42\n\n",
           all_correct ? "✅" : "❌", N);

    HIP_CHECK(hipFree(d_output));
    return all_correct ? 0 : 1;
}
