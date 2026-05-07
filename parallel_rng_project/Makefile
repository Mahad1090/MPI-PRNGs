# ==============================================================================
# File: Makefile
# Project: Parallel Random Number Generation — CS-3006 SP 2026
# Paper: Salmon et al., Parallel Random Numbers: As Easy as 1,2,3, SC11
#
#  make compile              build all CPU targets
#  make compile-gpu          build CUDA target (needs nvcc)
#  make run                  run all local experiments in order
#  make run-mpi-2            MPI 2 local processes
#  make deploy               copy source to remote machine
#  make remote-compile       compile mpi_threefry on remote
#  make run-mpi-multi-4      4 ranks across both machines
#  make help                 full target list
# ==============================================================================

ROOT        := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
RESULTS_DIR := $(ROOT)/results
PLOTS_DIR   := $(ROOT)/plots
R123_INC    := $(ROOT)/random123/include

CXX      := g++
MPICXX   := mpic++
NVCC     := nvcc
CXXFLAGS := -O2 -std=c++17

BASELINE_BIN := $(ROOT)/baseline/sequential_threefry
MPI_BIN      := $(ROOT)/cpu_parallel/mpi_threefry
GPU_BIN      := $(ROOT)/gpu_parallel/cuda_philox
BW_BIN       := $(ROOT)/hardware_profiling/bandwidth_test
ROOF_BIN     := $(ROOT)/hardware_profiling/roofline_analysis

# Override on command line:  make run-mpi-4 N=50000000
N    := 10000000
SEED := 42

# Remote machine — adjust REMOTE_HOME if needed
REMOTE       := rayyan@172.15.70.230
REMOTE_HOME  := /home/rayyan
REMOTE_DIR   := $(REMOTE_HOME)/parallel_rng_project
REMOTE_BIN   := $(REMOTE_DIR)/cpu_parallel/mpi_threefry
LOCAL_SLOTS  := 10
REMOTE_SLOTS := 8
APPFILE      := $(ROOT)/mpi_appfile
HOSTFILE     := $(ROOT)/hostfile

.PHONY: all compile compile-baseline compile-mpi compile-gpu compile-bw compile-roof \
        run run-baseline run-mpi-1 run-mpi-2 run-mpi-4 run-mpi-8 run-mpi-10 \
        run-mpi-multi-2 run-mpi-multi-4 run-mpi-multi-8 run-mpi-multi-16 run-mpi-multi-18 \
        run-gpu run-bw run-roof run-plots \
        deploy remote-compile sync-results hostfile run-remote clean clean-results help

# ------------------------------------------------------------------------------
all: compile

# ------------------------------------------------------------------------------
# COMPILATION
# ------------------------------------------------------------------------------
compile: compile-baseline compile-mpi compile-bw compile-roof
	@echo "  All CPU targets compiled."

compile-baseline: $(BASELINE_BIN)
compile-mpi:      $(MPI_BIN)
compile-bw:       $(BW_BIN)
compile-roof:     $(ROOF_BIN)
compile-gpu:      $(GPU_BIN)

$(BASELINE_BIN): $(ROOT)/baseline/sequential_threefry.cpp
	@echo "  Compiling baseline/sequential_threefry.cpp..."
	$(CXX) $(CXXFLAGS) -I$(R123_INC) -o $@ $<
	@echo "  OK  baseline"

$(MPI_BIN): $(ROOT)/cpu_parallel/mpi_threefry.cpp
	@echo "  Compiling cpu_parallel/mpi_threefry.cpp..."
	$(MPICXX) $(CXXFLAGS) -I$(R123_INC) -o $@ $<
	@echo "  OK  mpi_threefry"

$(GPU_BIN): $(ROOT)/gpu_parallel/cuda_philox.cu
	@echo "  Compiling gpu_parallel/cuda_philox.cu..."
	$(NVCC) $(CXXFLAGS) -I$(R123_INC) -o $@ $<
	@echo "  OK  cuda_philox"

$(BW_BIN): $(ROOT)/hardware_profiling/bandwidth_test.cpp
	@echo "  Compiling hardware_profiling/bandwidth_test.cpp..."
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "  OK  bandwidth_test"

$(ROOF_BIN): $(ROOT)/hardware_profiling/roofline_analysis.cpp
	@echo "  Compiling hardware_profiling/roofline_analysis.cpp..."
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "  OK  roofline_analysis"

# ------------------------------------------------------------------------------
# LOCAL RUN TARGETS
# All programs run from ROOT so 'results/' resolves correctly.
# ------------------------------------------------------------------------------
$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

run: run-baseline run-mpi-1 run-mpi-2 run-mpi-4 run-mpi-8 run-mpi-10 run-bw run-roof run-plots
	@echo ""
	@echo "  All local experiments complete."

run-baseline: $(BASELINE_BIN) | $(RESULTS_DIR)
	@echo "  -- Single Core Threefry Baseline"
	cd $(ROOT) && $(BASELINE_BIN) $(N)

run-mpi-1: $(MPI_BIN) | $(RESULTS_DIR)
	@echo "  -- MPI Threefry: 1 process"
	cd $(ROOT) && mpirun --oversubscribe -np 1 $(MPI_BIN) $(N) $(SEED)

run-mpi-2: $(MPI_BIN) | $(RESULTS_DIR)
	@echo "  -- MPI Threefry: 2 processes"
	cd $(ROOT) && mpirun --oversubscribe -np 2 $(MPI_BIN) $(N) $(SEED)

run-mpi-4: $(MPI_BIN) | $(RESULTS_DIR)
	@echo "  -- MPI Threefry: 4 processes"
	cd $(ROOT) && mpirun --oversubscribe -np 4 $(MPI_BIN) $(N) $(SEED)

run-mpi-8: $(MPI_BIN) | $(RESULTS_DIR)
	@echo "  -- MPI Threefry: 8 processes"
	cd $(ROOT) && mpirun --oversubscribe -np 8 $(MPI_BIN) $(N) $(SEED)

# 10 processes (matches localhost slots=10 in hosts.txt)
run-mpi-10: $(MPI_BIN) | $(RESULTS_DIR)
	@echo "  -- MPI Threefry: 10 processes"
	cd $(ROOT) && mpirun --oversubscribe -np 10 $(MPI_BIN) $(N) $(SEED)

run-gpu: $(GPU_BIN) | $(RESULTS_DIR)
	@echo "  -- GPU Philox CUDA"
	cd $(ROOT) && $(GPU_BIN) $(N)

run-bw: $(BW_BIN) | $(RESULTS_DIR)
	@echo "  -- Hardware Bandwidth & Compute Profiling"
	cd $(ROOT) && $(BW_BIN)

run-roof: $(ROOF_BIN) | $(RESULTS_DIR)
	@echo "  -- Roofline & Performance Summary"
	cd $(ROOT) && $(ROOF_BIN)

run-plots: | $(RESULTS_DIR)
	@echo "  -- Generating Plots"
	python3 $(ROOT)/plots/generate_plots.py

# Run all multi-node MPI experiments + hardware profiling + plots
# Prerequisite: make deploy && make remote-compile
run-remote: run-mpi-multi-2 run-mpi-multi-4 run-mpi-multi-8 run-mpi-multi-16 run-mpi-multi-18 run-bw run-roof run-plots
	@echo ""
	@echo "  All remote experiments complete."
	@echo "  Ranks distributed: localhost=10 slots, 172.15.70.230=8 slots"

# ------------------------------------------------------------------------------
# MULTI-NODE MPI  (local + rayyan@172.15.70.230)
#
# Uses an MPI appfile so each host references its own binary path.
# Prerequisite:  make deploy && make remote-compile
# Rank 0 = localhost  =>  mpi_results.csv written locally.
# ------------------------------------------------------------------------------
hostfile:
	@printf "localhost slots=$(LOCAL_SLOTS)\n" > $(HOSTFILE)
	@printf "172.15.70.230 slots=$(REMOTE_SLOTS)\n" >> $(HOSTFILE)
	@echo "  OK  $(HOSTFILE):"
	@cat $(HOSTFILE)

# 2 ranks: 1 local + 1 remote
run-mpi-multi-2: $(MPI_BIN) | $(RESULTS_DIR)
	@echo "  -- Multi-node MPI: 1 local + 1 remote"
	@printf "%s\n" \
	    "-H localhost -np 1 $(MPI_BIN) $(N) $(SEED)" \
	    "-H 172.15.70.230 -np 1 $(REMOTE_BIN) $(N) $(SEED)" \
	    > $(APPFILE)
	cd $(ROOT) && mpirun --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --app $(APPFILE)

# 4 ranks: 2 local + 2 remote
run-mpi-multi-4: $(MPI_BIN) | $(RESULTS_DIR)
	@echo "  -- Multi-node MPI: 2 local + 2 remote"
	@printf "%s\n" \
	    "-H localhost -np 2 $(MPI_BIN) $(N) $(SEED)" \
	    "-H 172.15.70.230 -np 2 $(REMOTE_BIN) $(N) $(SEED)" \
	    > $(APPFILE)
	cd $(ROOT) && mpirun --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --app $(APPFILE)

# 8 ranks: 4 local + 4 remote  (evenly split)
run-mpi-multi-8: $(MPI_BIN) | $(RESULTS_DIR)
	@echo "  -- Multi-node MPI: 4 local + 4 remote"
	@printf "%s\n" \
	    "-H localhost -np 4 $(MPI_BIN) $(N) $(SEED)" \
	    "-H 172.15.70.230 -np 4 $(REMOTE_BIN) $(N) $(SEED)" \
	    > $(APPFILE)
	cd $(ROOT) && mpirun --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --app $(APPFILE)

# 16 ranks: 8 local + 8 remote  (max even split — limited by remote=8 slots)
run-mpi-multi-16: $(MPI_BIN) | $(RESULTS_DIR)
	@echo "  -- Multi-node MPI: 8 local + 8 remote"
	@printf "%s\n" \
	    "-H localhost -np 8 $(MPI_BIN) $(N) $(SEED)" \
	    "-H 172.15.70.230 -np 8 $(REMOTE_BIN) $(N) $(SEED)" \
	    > $(APPFILE)
	cd $(ROOT) && mpirun --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --app $(APPFILE)

# 18 ranks: 10 local + 8 remote  (FULL CAPACITY — localhost=10, remote=8)
run-mpi-multi-18: $(MPI_BIN) | $(RESULTS_DIR)
	@echo "  -- Multi-node MPI: 10 local + 8 remote (FULL CAPACITY)"
	@printf "%s\n" \
	    "-H localhost -np 10 $(MPI_BIN) $(N) $(SEED)" \
	    "-H 172.15.70.230 -np 8 $(REMOTE_BIN) $(N) $(SEED)" \
	    > $(APPFILE)
	cd $(ROOT) && mpirun --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --app $(APPFILE)

# ------------------------------------------------------------------------------
# REMOTE MACHINE SETUP
# ------------------------------------------------------------------------------

# Copy mpi_threefry.cpp + Random123 headers to remote
deploy:
	@echo "  Deploying to $(REMOTE):$(REMOTE_DIR)..."
	ssh $(REMOTE) "mkdir -p $(REMOTE_DIR)/cpu_parallel $(REMOTE_DIR)/random123/include $(REMOTE_DIR)/results"
	scp $(ROOT)/cpu_parallel/mpi_threefry.cpp $(REMOTE):$(REMOTE_DIR)/cpu_parallel/mpi_threefry.cpp
	rsync -avz --delete $(ROOT)/random123/include/ $(REMOTE):$(REMOTE_DIR)/random123/include/
	@echo "  OK  deploy done"

# Compile mpi_threefry on the remote
remote-compile: deploy
	@echo "  Compiling mpi_threefry on $(REMOTE)..."
	ssh $(REMOTE) "mpic++ -O2 -std=c++17 -I$(REMOTE_DIR)/random123/include -o $(REMOTE_DIR)/cpu_parallel/mpi_threefry $(REMOTE_DIR)/cpu_parallel/mpi_threefry.cpp && echo '  OK  remote compile done'"

# Pull CSVs from remote (rank 0 is local so usually not needed)
sync-results:
	scp $(REMOTE):$(REMOTE_DIR)/results/*.csv $(RESULTS_DIR)/ || true
	@echo "  OK  results synced"

# ------------------------------------------------------------------------------
# CLEAN
# ------------------------------------------------------------------------------
clean:
	rm -f $(BASELINE_BIN) $(MPI_BIN) $(GPU_BIN) $(BW_BIN) $(ROOF_BIN)
	rm -f $(HOSTFILE) $(APPFILE)
	@echo "  OK  binaries removed"

clean-results:
	rm -f $(RESULTS_DIR)/*.csv $(RESULTS_DIR)/*.txt
	@echo "  OK  results cleared"

# ------------------------------------------------------------------------------
# HELP
# ------------------------------------------------------------------------------
help:
	@echo ""
	@echo "  COMPILATION"
	@echo "    make compile              build all CPU targets"
	@echo "    make compile-gpu          build CUDA target (needs nvcc)"
	@echo "    make compile-baseline"
	@echo "    make compile-mpi"
	@echo ""
	@echo "  LOCAL RUNS"
	@echo "    make run                  full pipeline (baseline>mpi>bw>roof>plots)"
	@echo "    make run-baseline         single-core Threefry"
	@echo "    make run-mpi-1/2/4/8/10   N local MPI processes (10 = full LOCAL_SLOTS)"
	@echo "    make run-gpu              GPU Philox CUDA"
	@echo "    make run-bw               hardware bandwidth test"
	@echo "    make run-roof             roofline analysis"
	@echo "    make run-plots            generate all 7 plots"
	@echo ""
	@echo "  MULTI-NODE MPI  (run: make deploy && make remote-compile first)"
	@echo "    make run-mpi-multi-2      2 ranks  (1 local + 1 remote)"
	@echo "    make run-mpi-multi-4      4 ranks  (2 local + 2 remote)"
	@echo "    make run-mpi-multi-8    8 ranks  (4 local + 4 remote)"
	@echo "    make run-mpi-multi-16  16 ranks  (8 local + 8 remote, max even split)"
	@echo "    make run-mpi-multi-18  18 ranks  (10 local + 8 remote, FULL CAPACITY)"
	@echo ""
	@echo "  REMOTE SETUP"
	@echo "    make deploy               rsync source to rayyan@172.15.70.230"
	@echo "    make remote-compile       compile on remote"
	@echo "    make sync-results         pull CSVs from remote"
	@echo ""
	@echo "  OVERRIDES (pass on command line)"
	@echo "    make run-mpi-4 N=50000000"
	@echo "    make run-mpi-multi-8 LOCAL_SLOTS=2 REMOTE_SLOTS=6"
	@echo ""
