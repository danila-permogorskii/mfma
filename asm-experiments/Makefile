# ═══════════════════════════════════════════════════════════════════════════════
# AMDGPU Assembly Experiments — Top-Level Makefile
# AMDGPU Assembler-experiment — Toppnivå-Makefile
#
# This Makefile builds standalone GPU kernels written in AMDGPU assembly,
# links them into HSACO code objects, and compiles host programs to run them.
#
# Denna Makefile bygger fristående GPU-kärnor skrivna i AMDGPU-assembler,
# länkar dem till HSACO-kodobjekt och kompilerar värdprogram för att köra dem.
#
# Usage / Användning:
#   make help          Show all targets / Visa alla mål
#   make verify        Run environment verification / Kör miljöverifiering
#   make asm-01        Build experiment 01 / Bygg experiment 01
#   make all           Build everything / Bygg allt
#   make clean         Remove all build artifacts / Ta bort allt byggt
#
# ═══════════════════════════════════════════════════════════════════════════════

# ── ROCm Configuration ────────────────────────────────────────────────────────
# ── ROCm-konfiguration ────────────────────────────────────────────────────────
#
# ROCM_PATH: Base path to the ROCm installation.
# Override with: make ROCM_PATH=/opt/rocm-7.1.0 asm-01
# ROCM_PATH: Bassökväg till ROCm-installationen.
ROCM_PATH ?= /opt/rocm
LLVM      := $(ROCM_PATH)/llvm/bin
ARCH      := gfx942

# ── Assembly Toolchain ────────────────────────────────────────────────────────
# ── Assembler-verktygskedja ───────────────────────────────────────────────────
#
# CLANG_ASM:  Assembles .s → .o (ELF object)
#             Assemblerar .s → .o (ELF-objekt)
#
# LLD:        Links .o → .co (HSACO shared library)
#             Länkar .o → .co (HSACO delat bibliotek)
#
# OBJDUMP:    Disassembles .co for verification
#             Disassemblerar .co för verifiering
#
# READELF:    Reads ELF headers and kernel descriptor metadata
#             Läser ELF-huvuden och kärnbeskrivningsmetadata
CLANG_ASM := $(LLVM)/clang
LLD       := $(LLVM)/ld.lld
OBJDUMP   := $(LLVM)/llvm-objdump
READELF   := $(LLVM)/llvm-readelf

# Assembly flags: target triple + architecture
# Assemblerflaggor: måltrippel + arkitektur
ASMFLAGS  := -x assembler -target amdgcn-amd-amdhsa -mcpu=$(ARCH) -c

# Linker flags: produce shared library (HSACO format)
# Länkflaggor: producera delat bibliotek (HSACO-format)
LDFLAGS   := -shared

# ── Host Compiler ─────────────────────────────────────────────────────────────
# ── Värdkompilator ────────────────────────────────────────────────────────────
#
# hipcc compiles the C++ host code that loads and launches assembly kernels.
# hipcc kompilerar C++-värdkoden som laddar och startar assemblerkärnor.
HIPCC     := hipcc
HIPFLAGS  := -std=c++17 -O2

# ── Experiment Definitions ────────────────────────────────────────────────────
# ── Experimentdefinitioner ────────────────────────────────────────────────────
#
# Each experiment has:
#   - A .s assembly source  (kernel)
#   - A .cpp host source    (loader + launcher)
#   - Produces: .o, .co, and a host binary
#
# Varje experiment har:
#   - En .s-assemblerkälla  (kärna)
#   - En .cpp-värdkälla     (laddare + startare)
#   - Producerar: .o, .co och en värdexekverbar
EXPERIMENTS := asm-01 asm-02 asm-03 asm-04 asm-05 asm-06

.PHONY: all clean help verify $(EXPERIMENTS)

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo "AMDGPU Assembly Experiments — Build System"
	@echo "AMDGPU Assembler-experiment — Byggsystem"
	@echo "════════════════════════════════════════════"
	@echo ""
	@echo "Targets / Mål:"
	@echo "  verify   — Check environment (Kontrollera miljö)"
	@echo "  asm-01   — Hello Assembly (Hej Assembler)"
	@echo "  asm-02   — Vector Addition (Vektoraddition)"
	@echo "  asm-03   — LDS Manual Control (Manuell LDS-kontroll)"
	@echo "  asm-04   — MFMA Bare Metal (MFMA ren metall)"
	@echo "  asm-05   — MFMA Pipeline (MFMA-pipeline)"
	@echo "  asm-06   — Hand vs Compiler (Hand mot kompilator)"
	@echo "  all      — Build everything (Bygg allt)"
	@echo "  clean    — Remove artifacts (Ta bort artefakter)"
	@echo ""
	@echo "Inspection / Inspektion:"
	@echo "  make disasm-01  — Disassemble asm-01 code object"
	@echo "  make meta-01    — Show asm-01 kernel metadata"
	@echo ""
	@echo "Configuration / Konfiguration:"
	@echo "  ROCM_PATH=$(ROCM_PATH)"
	@echo "  ARCH=$(ARCH)"

# ── Environment Verification ──────────────────────────────────────────────────
verify:
	@chmod +x verify_asm_env.sh
	@./verify_asm_env.sh

# ═══════════════════════════════════════════════════════════════════════════════
# GENERIC BUILD RULES
# GENERISKA BYGGREGLER
#
# Pattern rules for the three-step assembly build pipeline:
# Mönsterregler för trestegs assembler-byggpipelinen:
#
#   .s → .o  (assemble / assemblera)
#   .o → .co (link / länka)
#   .cpp + .co → binary (compile host + embed path / kompilera värd)
# ═══════════════════════════════════════════════════════════════════════════════

# Step 1: Assemble .s to .o
# Steg 1: Assemblera .s till .o
%.o: %.s
	@echo "  [ASM]  $< → $@"
	$(CLANG_ASM) $(ASMFLAGS) -o $@ $<

# Step 2: Link .o to .co (HSACO)
# Steg 2: Länka .o till .co (HSACO)
%.co: %.o
	@echo "  [LINK] $< → $@"
	$(LLD) $(LDFLAGS) -o $@ $<

# ═══════════════════════════════════════════════════════════════════════════════
# PER-EXPERIMENT BUILD TARGETS
# BYGGMÅL PER EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

# ── asm-01: Hello Assembly ────────────────────────────────────────────────────
asm-01: asm-01-hello-asm/hello_asm.co asm-01-hello-asm/runner
	@echo "  ✅ asm-01 built (asm-01 byggt)"

asm-01-hello-asm/runner: asm-01-hello-asm/host.cpp asm-01-hello-asm/hello_asm.co
	@echo "  [HOST] $< → $@"
	$(HIPCC) $(HIPFLAGS) -o $@ $<

# ── asm-02: Vector Addition ───────────────────────────────────────────────────
asm-02: asm-02-vector-add/vector_add.co asm-02-vector-add/runner
	@echo "  ✅ asm-02 built (asm-02 byggt)"

asm-02-vector-add/runner: asm-02-vector-add/host.cpp asm-02-vector-add/vector_add.co
	@echo "  [HOST] $< → $@"
	$(HIPCC) $(HIPFLAGS) -o $@ $<

# ── asm-03: LDS Manual Control ────────────────────────────────────────────────
asm-03: asm-03-lds-manual/lds_reduce.co asm-03-lds-manual/runner
	@echo "  ✅ asm-03 built (asm-03 byggt)"

asm-03-lds-manual/runner: asm-03-lds-manual/host.cpp asm-03-lds-manual/lds_reduce.co
	@echo "  [HOST] $< → $@"
	$(HIPCC) $(HIPFLAGS) -o $@ $<

# ── asm-04: MFMA Bare Metal ───────────────────────────────────────────────────
asm-04: asm-04-mfma-bare/mfma_bare.co asm-04-mfma-bare/runner
	@echo "  ✅ asm-04 built (asm-04 byggt)"

asm-04-mfma-bare/runner: asm-04-mfma-bare/host.cpp asm-04-mfma-bare/mfma_bare.co
	@echo "  [HOST] $< → $@"
	$(HIPCC) $(HIPFLAGS) -o $@ $<

# ── asm-05: MFMA Pipeline ────────────────────────────────────────────────────
asm-05: asm-05-mfma-pipeline/mfma_pipeline.co asm-05-mfma-pipeline/runner
	@echo "  ✅ asm-05 built (asm-05 byggt)"

asm-05-mfma-pipeline/runner: asm-05-mfma-pipeline/host.cpp asm-05-mfma-pipeline/mfma_pipeline.co
	@echo "  [HOST] $< → $@"
	$(HIPCC) $(HIPFLAGS) -o $@ $<

# ── asm-06: Hand vs Compiler ──────────────────────────────────────────────────
asm-06: asm-06-mfma-vs-compiler/mfma_hand.co asm-06-mfma-vs-compiler/runner
	@echo "  ✅ asm-06 built (asm-06 byggt)"

asm-06-mfma-vs-compiler/runner: asm-06-mfma-vs-compiler/host.cpp asm-06-mfma-vs-compiler/mfma_hand.co
	@echo "  [HOST] $< → $@"
	$(HIPCC) $(HIPFLAGS) -o $@ $<

# ═══════════════════════════════════════════════════════════════════════════════
# INSPECTION TARGETS — for verifying assembly output
# INSPEKTIONSMÅL — för att verifiera assemblerutdata
# ═══════════════════════════════════════════════════════════════════════════════

disasm-%: asm-%-*//*.co
	@echo "── Disassembly / Disassemblering ──"
	$(OBJDUMP) -d $<

meta-%: asm-%-*//*.co
	@echo "── Kernel Metadata / Kärnmetadata ──"
	$(READELF) -n $<

sections-%: asm-%-*//*.co
	@echo "── ELF Sections / ELF-sektioner ──"
	$(READELF) -S -s $<

# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATE TARGETS
# AGGREGATMÅL
# ═══════════════════════════════════════════════════════════════════════════════

all: $(EXPERIMENTS)
	@echo ""
	@echo "═══════════════════════════════════════════════════════"
	@echo "  ✅ All assembly experiments built successfully"
	@echo "  ✅ Alla assembler-experiment byggda framgångsrikt"
	@echo "═══════════════════════════════════════════════════════"

clean:
	find . -name '*.o' -o -name '*.co' -o -name 'runner' | xargs rm -f
	@echo "  ✅ Cleaned / Städat"
