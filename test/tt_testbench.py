"""
tt_testbench.py — cocotb driver + reference model for tt_um_koderchina_convCore
==============================================================================

Protocol driver and golden model only. The @cocotb.test() entry points live in
test.py, which is what COCOTB_TEST_MODULES names; this module is imported by it.

Every `dut` below is the tb.v wrapper (TOPLEVEL = tb), not the design itself, so
`dut.ui_in` / `dut.uio_out` / ... are tb's own regs and wires. tb.v deliberately
mirrors the tt_um_* port names, which is why the driver needs no adaptation.

Drives the design through its TinyTapeout pins:

  ui_in[7:0]   shared data byte — TWO 4-bit values, low nibble first
  uio_in[0]    mode       0 = PROGRAM, 1 = STREAM
  uio_in[1]    in_valid   ui_in holds a byte this cycle
  uio_in[2]    out_ready  testbench can accept an output byte this cycle
  uio_out[3]   in_ready   DUT can accept a byte on ui_in this cycle
  uio_out[4]   out_valid  uo_out holds a valid pooled-output byte
  uio_out[5]   busy       1 whenever mid-frame

ONE OUTPUT CHANNEL PER PASS
---------------------------
An output channel is an independent full convolution over all input channels,
so the chip computes exactly one of them per pass. The HOST owns the
output-channel loop: reprogram the weights and bias, re-stream the whole input
image, collect that channel's pooled feature map, repeat. Stacking the maps is
off-chip work. test_top_multi_output_channel_passes below is the end-to-end
proof of that protocol.

PROGRAM image (nibble addresses, written in order from 0 — the address never
crosses the pins, the DUT counts internally):

  nibble  0.. 19   config  H_IN, W_IN, IN_CHANNELS, PAD_W, PAD_H
                           4 nibbles each, little-endian
  nibble 20..      weights addr = 20 + ic*9 + kw*3 + tap
  nibble ..        bias    a single register

Region boundaries come from the compile-time MAX_* envelope, not the
programmed channel count, so the image is always TOTAL_REGS nibbles and unused
channel slots are simply written with don't-care values. TOTAL_REGS is ODD, so
the final pin byte carries one real nibble plus a don't-care one; the DUT gates
that extra nibble off because prog_addr has already reached TOTAL_REGS.

STREAM order is column-major, input-channel in the middle, row innermost:

  for col: for ic: for row in 0,2,..H_IN-2:
      byte = {pix[row+1][ic][col], pix[row][ic][col]}

Output is produced WHILE input is still arriving, so the drain runs
concurrently in a background monitor rather than as a separate phase. Each
pooled value is one output byte: the value in [3:0], zero in [7:4].
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly, Timer
import os
import random

STEP = Timer(1, units="step")

# Clock period in ns. RTL sim is zero-delay, so 10ns is plenty. The gate-level
# netlist is compiled with `UNIT_DELAY #1 -- 1ns per CELL, not real cell timing
# -- and its deepest combinational paths run tens of cells, so it needs a period
# longer than that depth or logic is sampled before it settles. This is a
# simulation artifact only: STA signs the design off at CLOCK_PERIOD=1000ns with
# ~598ns of setup slack.
CLK_PERIOD_NS = int(os.environ.get("CLK_PERIOD_NS", "10"))

# How long after a clock edge the DUT's outputs are sampled.
#
# ReadOnly() runs once delta-cycle activity has settled -- but inertial (#N)
# delays advance simulation TIME, so they have NOT elapsed at that point. At RTL
# that distinction is invisible, because combinational logic resolves in deltas.
# In a gate-level netlist compiled with `UNIT_DELAY every cell costs 1ns, so the
# path from a flop to a pin resolves at edge+Nns and a sample taken at edge+0
# returns the PREVIOUS cycle's value. The whole ready/valid handshake then runs
# one cycle out of phase: bytes get injected a cycle early and drained bytes go
# missing.
#
# That failure is independent of the clock period (the sample point is pinned to
# the edge, not to the end of the cycle), which is why slowing the clock from
# 10ns to 2000ns produced byte-identical wrong results. Waiting most of a cycle
# before sampling is what actually fixes it, and costs nothing at RTL.
#
# The window must exceed the deepest flop-to-pin combinational delay -- one ns
# per cell under `UNIT_DELAY -- so gate-level runs also need a clock period long
# enough to contain it. See GL_CLK_PERIOD_NS in the Makefile.
SETTLE = Timer(max(1, CLK_PERIOD_NS * 4 // 5), units="ns")

# ---- Compile-time envelope (must match the Makefile's MAX_* values) ---------
DATA_WIDTH      = 4
K               = 3
MAX_IN_CHANNELS = 4

CFG_NIBBLES = 20
BIAS_ADDR   = CFG_NIBBLES + MAX_IN_CHANNELS * K * K
TOTAL_REGS  = BIAS_ADDR + 1

# ---- Runtime geometry for these tests --------------------------------------
H_IN        = 8      # MUST be even — a pin byte carries two rows
W_IN        = 8
IN_CHANNELS = 2
PAD_W       = 1
PAD_H       = 1

# Host-side only: how many output channels the "MCU" collects by re-streaming.
NUM_OUT_CHANNELS = 3

PADDED_H = H_IN + 2 * PAD_H
PADDED_W = W_IN + 2 * PAD_W
OUT_H    = PADDED_H - K + 1
OUT_W    = PADDED_W - K + 1
POOL_H   = OUT_H // 2
POOL_W   = OUT_W // 2

OUT_BYTES = POOL_H * POOL_W        # one byte per pooled value

INT_MIN = -(1 << (DATA_WIDTH - 1))     # -8
INT_MAX = (1 << (DATA_WIDTH - 1)) - 1  # +7
NMASK   = (1 << DATA_WIDTH) - 1

# uio_in bit positions (testbench → DUT)
UIO_IN_MODE      = 0
UIO_IN_VALID     = 1
UIO_IN_OUT_READY = 2

# uio_out bit positions (DUT → testbench)
UIO_OUT_IN_READY  = 3
UIO_OUT_OUT_VALID = 4
UIO_OUT_BUSY      = 5

MODE_PROGRAM = 0
MODE_STREAM  = 1


def to_signed(raw):
    raw &= NMASK
    return raw - (1 << DATA_WIDTH) if raw >= (1 << (DATA_WIDTH - 1)) else raw


def saturate(x):
    return max(INT_MIN, min(INT_MAX, x))


# =============================================================================
# Reference model — one Conv -> bias -> ReLU -> MaxPool -> saturate stage
# =============================================================================

def ref_pad(feature_maps, pad_h, pad_w):
    out = []
    for fm in feature_maps:
        h, w = len(fm), len(fm[0])
        p = [[0] * (w + 2 * pad_w) for _ in range(h + 2 * pad_h)]
        for r in range(h):
            for c in range(w):
                p[r + pad_h][c + pad_w] = fm[r][c]
        out.append(p)
    return out


def _ref_pe_outputs(weights, column):
    """Mirrors pe.v: two-deep history, first K-1 samples produce nothing."""
    hist, hvalid, out = [0, 0], 0, []
    for i, sample in enumerate(column):
        if i == 0:
            out.append(0)
            hist, hvalid = [sample, 0], 1
        else:
            if hvalid == 3:
                out.append(sample * weights[2] + hist[0] * weights[1] + hist[1] * weights[0])
            else:
                out.append(0)
            hist = [sample, hist[0]]
            hvalid = min((hvalid << 1 | 1) & 3, 3)
    return out


def ref_conv_2d(input_map, kernel):
    rows, cols = len(input_map), len(input_map[0])
    h_out, w_out = rows - K + 1, cols - K + 1
    output = [[0] * w_out for _ in range(h_out)]
    for kw in range(K):
        weights = [kernel[kh][kw] for kh in range(K)]
        for j in range(cols):
            out_j = j - kw
            if 0 <= out_j < w_out:
                col_vals = [input_map[r][j] for r in range(rows)]
                pe_out = _ref_pe_outputs(weights, col_vals)
                for i in range(h_out):
                    output[i][out_j] += pe_out[i + K - 1]
    return output


def ref_maxpool(fm):
    H, W = len(fm), len(fm[0])
    hp, wp = H // 2, W // 2
    out = [[0] * wp for _ in range(hp)]
    for pr in range(hp):
        for pc in range(wp):
            out[pr][pc] = max(fm[pr * 2 + dr][pc * 2 + dc]
                              for dr in range(2) for dc in range(2))
    return out


def ref_conv_layer(input_maps, kernels, bias):
    """kernels[ic][kh][kw], one bias -> ONE pooled, saturated feature map."""
    padded = ref_pad(input_maps, PAD_H, PAD_W)
    h_out = len(padded[0]) - K + 1
    w_out = len(padded[0][0]) - K + 1
    total = [[0] * w_out for _ in range(h_out)]
    for ic in range(IN_CHANNELS):
        partial = ref_conv_2d(padded[ic], kernels[ic])
        for r in range(h_out):
            for c in range(w_out):
                total[r][c] += partial[r][c]
    biased = [[total[r][c] + bias for c in range(w_out)] for r in range(h_out)]
    relu_out = [[max(0, x) for x in row] for row in biased]
    return [[saturate(x) for x in row] for row in ref_maxpool(relu_out)]


def expected_values(input_maps, kernels, bias):
    """Values leave the core COLUMN-MAJOR: conv emits a whole column of output
    rows per input column, so pool_col is the outer index."""
    pooled = ref_conv_layer(input_maps, kernels, bias)
    return [pooled[pr][pc] for pc in range(POOL_W) for pr in range(POOL_H)]


# =============================================================================
# Register image
# =============================================================================

def build_register_image(kernels, bias):
    """Full nibble image, in the order the DUT's internal counter expects."""
    regs = [0] * TOTAL_REGS

    for f, val in enumerate([H_IN, W_IN, IN_CHANNELS, PAD_W, PAD_H]):
        for n in range(4):
            regs[f * 4 + n] = (val >> (DATA_WIDTH * n)) & NMASK

    for ic in range(IN_CHANNELS):
        for kw in range(K):
            for tap in range(K):
                kh = K - 1 - tap        # tap 0 multiplies the newest sample
                regs[CFG_NIBBLES + ic * K * K + kw * K + tap] = \
                    int(kernels[ic][kh][kw]) & NMASK

    regs[BIAS_ADDR] = int(bias) & NMASK
    return regs


def image_to_bytes(regs):
    """Pack nibbles two per byte, low nibble first. TOTAL_REGS is odd, so the
    image is padded with one don't-care nibble; the DUT drops it because
    prog_addr has already reached TOTAL_REGS by then."""
    padded = list(regs)
    if len(padded) % 2:
        padded.append(0)
    return [(padded[i] & NMASK) | ((padded[i + 1] & NMASK) << DATA_WIDTH)
            for i in range(0, len(padded), 2)]


def stream_bytes(input_maps):
    """Column-major, ic in the middle, two rows per byte."""
    out = []
    for col in range(W_IN):
        for ic in range(IN_CHANNELS):
            for row in range(0, H_IN, 2):
                lo = input_maps[ic][row][col] & NMASK
                hi = input_maps[ic][row + 1][col] & NMASK
                out.append(lo | (hi << DATA_WIDTH))
    return out


# =============================================================================
# Pin helpers
# =============================================================================

def pack_uio_in(mode=0, in_valid=0, out_ready=1):
    return (((mode & 1) << UIO_IN_MODE)
            | ((in_valid & 1) << UIO_IN_VALID)
            | ((out_ready & 1) << UIO_IN_OUT_READY))


class Pins:
    """Single owner of uio_in.

    mode/in_valid (driven by the sender) and out_ready (driven by the output
    monitor) share one pin byte, so both writers compose the WHOLE byte from
    this shared state. That way it does not matter which of them writes last
    within a cycle -- they always produce the same value.
    """

    def __init__(self, dut):
        self.dut       = dut
        self.mode      = MODE_PROGRAM
        self.in_valid  = 0
        self.out_ready = 1

    def drive(self):
        self.dut.uio_in.value = pack_uio_in(self.mode, self.in_valid,
                                            self.out_ready)

    def set(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.drive()


def uio_out_bit(dut, bit):
    return (int(dut.uio_out.value) >> bit) & 1


async def _send_byte(dut, pins, value, mode):
    """Drive one byte, waiting for the clock edge that actually consumes it.

    in_ready is combinational over registered DUT state (never depends on
    in_valid), so sampling it via RisingEdge+ReadOnly gives the value as
    updated BY that edge -- i.e. what gates consumption at the *next* edge,
    not the one just observed. Treating that as "already consumed" and moving
    on races ahead of the DUT and silently drops a byte at every handshake
    transition. So: sample ready via ReadOnly BEFORE the edge, and only treat
    the byte as sent once that edge has actually passed.

    Being combinational is also why the sample has to wait SETTLE first -- at
    gate level that path has not resolved at edge+0.
    """
    dut.ui_in.value = int(value) & 0xFF
    pins.set(mode=mode, in_valid=1)
    while True:
        await SETTLE
        await ReadOnly()
        ready = uio_out_bit(dut, UIO_OUT_IN_READY)
        await RisingEdge(dut.clk)
        if ready:
            break
    await STEP
    pins.set(in_valid=0)


class OutputMonitor:
    """Drives out_ready and collects every byte the handshake actually accepts.

    out_ready is set at the TOP of each cycle (before ReadOnly, which forbids
    writes) so the DUT's combinational serializer sees it the same cycle. A
    byte is consumed only on an edge where out_valid AND out_ready were both
    high beforehand -- with a stall pattern that is no longer every valid
    cycle, which is exactly what exercises the output FIFO.
    """

    def __init__(self, dut, pins, ready_pattern=None):
        self.dut     = dut
        self.pins    = pins
        self.bytes   = []
        self.pattern = ready_pattern or [1]
        self.i       = 0

    async def run(self):
        while True:
            self.pins.set(out_ready=self.pattern[self.i % len(self.pattern)])
            self.i += 1
            await SETTLE
            await ReadOnly()
            byte = None
            if uio_out_bit(self.dut, UIO_OUT_OUT_VALID) and self.pins.out_ready:
                byte = int(self.dut.uo_out.value) & 0xFF
            await RisingEdge(self.dut.clk)
            if byte is not None:
                self.bytes.append(byte)


def decode_values(byte_list):
    """Inverse of top.v's serializer: one pooled value per byte, low nibble."""
    for byte in byte_list:
        assert (byte >> DATA_WIDTH) == 0, (
            f"output byte 0x{byte:02x} has a non-zero high nibble"
        )
    return [to_signed(byte & NMASK) for byte in byte_list]


# =============================================================================
# DUT sequencing
# =============================================================================

async def reset_dut(dut, pins):
    if hasattr(dut, 'VPWR'):
        dut.VPWR.value = 1
        dut.VGND.value = 0
    dut.ena.value   = 1
    dut.ui_in.value = 0
    pins.set(mode=MODE_PROGRAM, in_valid=0)
    dut.rst_n.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await STEP


async def enter_program(dut, pins):
    """Drop mode and wait until the FSM has actually returned to PROGRAM.

    After a frame drains, the DUT re-enters STREAM (mode was still high when
    the last byte was accepted). Programming the next output channel's weights
    must not begin until busy has fallen, or the first register byte would be
    consumed as a stream sample instead.
    """
    pins.set(mode=MODE_PROGRAM, in_valid=0)
    for _ in range(200):
        await SETTLE
        await ReadOnly()
        in_prog = uio_out_bit(dut, UIO_OUT_BUSY) == 0
        await RisingEdge(dut.clk)
        if in_prog:
            await STEP
            return
    raise AssertionError("FSM never returned to PROGRAM")


async def program(dut, pins, kernels, bias):
    await enter_program(dut, pins)
    for byte in image_to_bytes(build_register_image(kernels, bias)):
        await _send_byte(dut, pins, byte, MODE_PROGRAM)
    dut._log.info("Programmed %d register nibbles", TOTAL_REGS)


async def stream_and_collect(dut, pins, input_maps, mon):
    """Raise mode to STREAM, push every input byte, then let the tail drain.

    Output is full-duplex with input -- most of the frame's bytes arrive
    WHILE input is still being sent, not after. The target must be captured
    against the baseline before any of this frame's bytes have gone out, or
    the bytes that already arrived during streaming get double-counted on
    top of themselves.
    """
    target = len(mon.bytes) + OUT_BYTES

    pins.set(mode=MODE_STREAM, in_valid=0)
    await RisingEdge(dut.clk)
    await STEP

    for byte in stream_bytes(input_maps):
        await _send_byte(dut, pins, byte, MODE_STREAM)

    pins.set(in_valid=0)

    for _ in range(400 * OUT_BYTES + 40000):
        if len(mon.bytes) >= target:
            break
        await RisingEdge(dut.clk)
    assert len(mon.bytes) >= target, (
        f"timed out draining: got {len(mon.bytes)}, wanted {target} bytes"
    )


def random_kernels(seed, lo=-2, hi=2):
    """kernels[ic][kh][kw] — one output channel's worth."""
    random.seed(seed)
    return [[[random.randint(lo, hi) for _ in range(K)] for _ in range(K)]
            for _ in range(IN_CHANNELS)]


def random_input(lo=-4, hi=4):
    return [[[random.randint(lo, hi) for _ in range(W_IN)] for _ in range(H_IN)]
            for _ in range(IN_CHANNELS)]


async def run_frame(dut, pins, kernels, bias, input_maps, mon, do_program=True):
    if do_program:
        await program(dut, pins, kernels, bias)
    before = len(mon.bytes)
    await stream_and_collect(dut, pins, input_maps, mon)
    got = decode_values(mon.bytes[before:before + OUT_BYTES])
    exp = expected_values(input_maps, kernels, bias)
    assert got == exp, (
        f"pooled feature map mismatch\n  got      {got}\n  expected {exp}"
    )
    return got


async def start_dut(dut, ready_pattern=None):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    pins = Pins(dut)
    await reset_dut(dut, pins)
    mon = OutputMonitor(dut, pins, ready_pattern)
    cocotb.start_soon(mon.run())
    return pins, mon
