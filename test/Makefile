# Makefile
# See https://docs.cocotb.org/en/stable/quickstart.html for more info

# defaults
SIM ?= icarus
FST ?= -fst # Use more efficient FST format
TOPLEVEL_LANG ?= verilog
SRC_DIR = $(PWD)/../src

# Must match `source_files` in info.yaml. Wrapper first, then top.v, then the
# modules below it. tt_um_koderchina_convCore.v opens with `default_nettype
# none and closes with `default_nettype wire, so it does not leak that setting
# into the files compiled after it.
PROJECT_SOURCES = \
	tt_um_koderchina_convCore.v \
	top.v \
	convCore.v \
	convLayer.v \
	conv.v \
	pad.v \
	pe.v \
	relu.v \
	maxpool.v \
	config_regfile.v \
	weight_regfile.v \
	weight_bank.v

ifneq ($(GATES),yes)

# RTL simulation:
SIM_BUILD				= sim_build/rtl
VERILOG_SOURCES += $(addprefix $(SRC_DIR)/,$(PROJECT_SOURCES))

else

# Gate level simulation:
SIM_BUILD				= sim_build/gl
COMPILE_ARGS    += -DGL_TEST
COMPILE_ARGS    += -DFUNCTIONAL
COMPILE_ARGS    += -DUSE_POWER_PINS
COMPILE_ARGS    += -DSIM
COMPILE_ARGS    += -DUNIT_DELAY=\#1
VERILOG_SOURCES += $(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/verilog/primitives.v
VERILOG_SOURCES += $(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/verilog/sky130_fd_sc_hd.v

# this gets copied in by the GDS action workflow
VERILOG_SOURCES += $(PWD)/gate_level_netlist.v

# The netlist is compiled with `UNIT_DELAY = #1, so every cell costs 1ns and the
# deepest flop-to-pin path runs tens of cells. The clock period has to contain
# that or the testbench samples outputs before they have settled and the
# ready/valid handshake runs a cycle out of phase. RTL is zero-delay and keeps
# tt_testbench.py's 10ns default.
export CLK_PERIOD_NS ?= 200

endif

# Allow sharing configuration between design and testbench via `include`:
COMPILE_ARGS 		+= -I$(SRC_DIR)

# Include the testbench sources:
VERILOG_SOURCES += $(PWD)/tb.v
TOPLEVEL = tb

# List test modules to run, separated by commas and without the .py suffix:
#
# Two names for the same thing: cocotb >=2.0 reads COCOTB_TEST_MODULES, cocotb
# 1.x reads MODULE. requirements.txt pins 2.0.1 (what CI installs), but the
# IIC-OSIC-TOOLS container ships 1.9.2 system-wide, and with only the 2.0 name
# set that container silently discovers no tests at all. Each version ignores
# the variable it does not know, so setting both runs everywhere.
COCOTB_TEST_MODULES = test
MODULE              = test

# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim
