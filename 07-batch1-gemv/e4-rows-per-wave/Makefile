# =============================================================================
# Experiment E4: Rows Per Wave — Makefile
# =============================================================================
#
# BUILD TARGETS:
#   make            — Build rows_per_wave_gemv
#   make run        — Build and run
#   make profile    — Run with rocprof (basic stats)
#   make profile-v3 — Run with rocprofv3 (hip-trace + kernel-trace)
#   make debug      — Build with -O0 -g -save-temps (for ISA inspection)
#   make isa        — Build with -save-temps and disassemble to $(TARGET).isa.txt
#   make clean      — Remove build artifacts
#
# =============================================================================

HIPCC = amdclang++
HIPFLAGS_LANG = -x hip
ARCH = gfx942
INCLUDES = -I../common
HIPFLAGS = $(HIPFLAGS_LANG) --offload-arch=$(ARCH) -O3 $(INCLUDES)
DEBUGFLAGS = $(HIPFLAGS_LANG) --offload-arch=$(ARCH) -O0 -g -save-temps $(INCLUDES)

TARGET = rows_per_wave_gemv
SRC = rows_per_wave_gemv.cpp

.PHONY: all run profile profile-v3 debug isa clean help

help:
	@echo "Experiment E4: Rows Per Wave"
	@echo "=============================================="
	@echo ""
	@echo "Targets:"
	@echo "  make            — Build $(TARGET)"
	@echo "  make run        — Build and run"
	@echo "  make profile    — Run with rocprof --stats"
	@echo "  make profile-v3 — Run with rocprofv3"
	@echo "  make debug      — Build with debug symbols + -save-temps"
	@echo "  make isa        — Build and disassemble to $(TARGET).isa.txt"
	@echo "  make clean      — Remove build artifacts"
	@echo ""

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(HIPFLAGS) -o $@ $<
	@echo "Built $(TARGET) for $(ARCH)"

run: $(TARGET)
	@echo ""
	@echo "Running $(TARGET)..."
	@echo ""
	./$(TARGET)

# rocprof --stats needs no external counters file, unlike module 06's
# profile target — that one references a metrics.txt that doesn't exist
# anywhere in this repo. This target works out of the box.
profile: $(TARGET)
	@echo ""
	@echo "Profiling with rocprof --stats..."
	@echo ""
	rocprof --stats ./$(TARGET)

profile-v3: $(TARGET)
	@echo ""
	@echo "Profiling with rocprofv3..."
	@echo ""
	rocprofv3 --hip-trace --kernel-trace -o results_v3 ./$(TARGET)
	@echo "Results saved to results_v3/"

debug: $(SRC)
	$(HIPCC) $(DEBUGFLAGS) -o $(TARGET)_debug $<
	@echo "Built $(TARGET)_debug with debug symbols and intermediate files"

isa: $(SRC)
	$(HIPCC) $(DEBUGFLAGS) -o $(TARGET)_isa $<
	llvm-objdump -d --offloading $(TARGET)_isa > $(TARGET).isa.txt
	@echo "Disassembly written to $(TARGET).isa.txt"

clean:
	rm -f $(TARGET) $(TARGET)_debug $(TARGET)_isa
	rm -f *.ll *.bc *.s *.o *.isa.txt
	rm -f *.csv results.stats.csv
	rm -rf results_v3/
	@echo "Cleaned build artifacts"
