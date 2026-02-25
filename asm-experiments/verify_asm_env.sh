#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# verify_asm_env.sh — AMDGPU Assembly Experiment Environment Checker
# verify_asm_env.sh — Kontroll av AMDGPU-assembler-experimentmiljö
#
# Run this script on your MI300X instance BEFORE starting any asm experiment.
# Kör detta skript på din MI300X-instans INNAN du börjar något asm-experiment.
#
# Usage / Användning:
#   chmod +x verify_asm_env.sh
#   ./verify_asm_env.sh
#
# Exit code 0 = all checks passed. Non-zero = something is missing.
# Avslutskod 0 = alla kontroller godkända. Icke-noll = något saknas.
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Colour codes (if terminal supports it) ────────────────────────────────────
# ── Färgkoder (om terminalen stödjer det) ─────────────────────────────────────
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[0;33m'
    CYAN='\033[0;36m'
    NC='\033[0m'
else
    GREEN='' RED='' YELLOW='' CYAN='' NC=''
fi

ROCM=${ROCM_PATH:-/opt/rocm}
PASS=0
FAIL=0
WARN=0

# ── Check function ────────────────────────────────────────────────────────────
# Runs a command silently. If it succeeds, prints ✅. If not, prints ❌.
# Kör ett kommando tyst. Om det lyckas, skriver ✅. Om inte, skriver ❌.
check() {
    local label="$1"
    local cmd="$2"
    if eval "$cmd" > /dev/null 2>&1; then
        printf "  ${GREEN}✅${NC} %s\n" "$label"
        ((PASS++))
    else
        printf "  ${RED}❌${NC} %s — ${RED}MISSING / SAKNAS${NC}\n" "$label"
        ((FAIL++))
    fi
}

warn_check() {
    local label="$1"
    local cmd="$2"
    if eval "$cmd" > /dev/null 2>&1; then
        printf "  ${GREEN}✅${NC} %s\n" "$label"
        ((PASS++))
    else
        printf "  ${YELLOW}⚠ ${NC} %s — ${YELLOW}OPTIONAL / VALFRITT${NC}\n" "$label"
        ((WARN++))
    fi
}

# ── Print version if tool exists ──────────────────────────────────────────────
# Skriv ut version om verktyget finns
print_version() {
    local tool_path="$1"
    local label="$2"
    if [ -x "$tool_path" ]; then
        local ver
        ver=$("$tool_path" --version 2>&1 | head -1)
        printf "    ${CYAN}→${NC} %s: %s\n" "$label" "$ver"
    fi
}

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  AMDGPU Assembly Experiment — Environment Verification"
echo "  AMDGPU Assembler-experiment — Miljöverifiering"
echo "═══════════════════════════════════════════════════════════════"
echo ""
printf "  ROCm path / ROCm-sökväg: ${CYAN}%s${NC}\n" "$ROCM"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# 1. GPU HARDWARE
# 1. GPU-HÅRDVARA
# ═══════════════════════════════════════════════════════════════════════════════
echo "── GPU Hardware / GPU-hårdvara ──"
check "rocminfo available"     "which rocminfo"
check "gfx942 GPU detected"    "rocminfo 2>/dev/null | grep -q 'gfx942'"

# Print GPU details if available
# Skriv ut GPU-detaljer om tillgängligt
if rocminfo > /dev/null 2>&1; then
    gpu_name=$(rocminfo 2>/dev/null | grep "Marketing Name" | head -1 | sed 's/.*: *//')
    if [ -n "$gpu_name" ]; then
        printf "    ${CYAN}→${NC} GPU: %s\n" "$gpu_name"
    fi
    cu_count=$(rocminfo 2>/dev/null | grep "Compute Unit:" | head -1 | sed 's/.*: *//')
    if [ -n "$cu_count" ]; then
        printf "    ${CYAN}→${NC} Compute Units: %s\n" "$cu_count"
    fi
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ASSEMBLY TOOLCHAIN
# 2. ASSEMBLER-VERKTYGSKEDJA
# ═══════════════════════════════════════════════════════════════════════════════
echo "── Assembly Toolchain / Assembler-verktygskedja ──"

check "clang (assembler frontend)"    "test -x ${ROCM}/llvm/bin/clang"
print_version "${ROCM}/llvm/bin/clang" "clang"

check "llvm-mc (machine code)"        "test -x ${ROCM}/llvm/bin/llvm-mc"
check "ld.lld (linker)"               "test -x ${ROCM}/llvm/bin/ld.lld"
check "llvm-objdump (disassembler)"   "test -x ${ROCM}/llvm/bin/llvm-objdump"
check "llvm-readelf (ELF reader)"     "test -x ${ROCM}/llvm/bin/llvm-readelf"

# Test that clang can actually assemble for gfx942
# Testa att clang faktiskt kan assemblera för gfx942
echo ""
echo "── Assembly Smoke Test / Assembler-röktest ──"
TMPDIR=$(mktemp -d)
cat > "${TMPDIR}/smoke.s" << 'EOF'
.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
.text
.globl smoke_test
.p2align 8
.type smoke_test,@function
smoke_test:
    s_endpgm
.Lsmoke_test_end:
    .size smoke_test, .Lsmoke_test_end - smoke_test

.rodata
.p2align 6
.amdhsa_kernel smoke_test
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 1
    .amdhsa_ieee_mode 1
    .amdhsa_wavefront_size32 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
EOF

check "clang assembles gfx942 .s file" \
    "${ROCM}/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx942 -c -o ${TMPDIR}/smoke.o ${TMPDIR}/smoke.s"

if [ -f "${TMPDIR}/smoke.o" ]; then
    check "ld.lld links to .co (HSACO)" \
        "${ROCM}/llvm/bin/ld.lld -shared -o ${TMPDIR}/smoke.co ${TMPDIR}/smoke.o"

    if [ -f "${TMPDIR}/smoke.co" ]; then
        check "llvm-objdump disassembles .co" \
            "${ROCM}/llvm/bin/llvm-objdump -d ${TMPDIR}/smoke.co 2>&1 | grep -q 's_endpgm'"
        check "llvm-readelf reads kernel descriptor" \
            "${ROCM}/llvm/bin/llvm-readelf -S ${TMPDIR}/smoke.co 2>&1 | grep -qE '(rodata|amdhsa)'"
    fi
fi
rm -rf "${TMPDIR}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# 3. HIP HOST TOOLS
# 3. HOST COMPILER
# 3. VÄRDKOMPILATOR
# ═══════════════════════════════════════════════════════════════════════════════
echo "── Host Compiler / Värdkompilator ──"
check "amdclang++ (host compiler)"    "test -x ${ROCM}/llvm/bin/amdclang++"
print_version "${ROCM}/llvm/bin/amdclang++" "amdclang++"
check "libamdhip64.so (runtime lib)"  "test -f ${ROCM}/lib/libamdhip64.so"

# Verify hipModuleLoad is available in the headers
# Verifiera att hipModuleLoad finns tillgängligt i header-filerna
check "hip/hip_runtime.h exists"       "test -f ${ROCM}/include/hip/hip_runtime.h"
check "hipModuleLoad in headers"       "grep -rq 'hipModuleLoad' ${ROCM}/include/hip/"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# 4. PROFILING
# 4. PROFILERING
# ═══════════════════════════════════════════════════════════════════════════════
echo "── Profiling Tools / Profileringsverktyg ──"
check "rocprofv3"                      "which rocprofv3"
warn_check "rocprof-compute"           "which rocprof-compute"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# 5. USEFUL EXTRAS
# 5. ANVÄNDBARA TILLÄGG
# ═══════════════════════════════════════════════════════════════════════════════
echo "── Useful Extras / Användbara tillägg ──"
warn_check "rocm-smi (GPU monitoring)"        "which rocm-smi"
warn_check "rocgdb (GPU debugger)"             "which rocgdb"
warn_check "amd_matrix_instruction_calculator" "which amd_matrix_instruction_calculator || test -f ${ROCM}/bin/amd_matrix_instruction_calculator"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# SAMMANFATTNING
# ═══════════════════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════════"
printf "  Results / Resultat: ${GREEN}%d passed${NC}" "$PASS"
if [ $FAIL -gt 0 ]; then
    printf ", ${RED}%d failed${NC}" "$FAIL"
fi
if [ $WARN -gt 0 ]; then
    printf ", ${YELLOW}%d optional warnings${NC}" "$WARN"
fi
echo ""
echo "═══════════════════════════════════════════════════════════════"

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "  ⚠  Some required components are missing."
    echo "     Vissa nödvändiga komponenter saknas."
    echo ""
    echo "     Ensure ROCm ≥ 6.0 is installed with the full LLVM toolchain:"
    echo "     Säkerställ att ROCm ≥ 6.0 är installerat med fullständig LLVM-verktygskedja:"
    echo ""
    echo "       sudo apt install rocm-llvm rocm-hip-sdk"
    echo ""
    echo "     Or set ROCM_PATH if installed in a non-standard location:"
    echo "     Eller ställ in ROCM_PATH om installerat på en icke-standard plats:"
    echo ""
    echo "       export ROCM_PATH=/opt/rocm-7.1.0"
    echo ""
    exit 1
else
    echo ""
    echo "  ✅ All required tools present. Ready to begin asm-01."
    echo "  ✅ Alla nödvändiga verktyg finns. Redo att börja asm-01."
    echo ""
    exit 0
fi
