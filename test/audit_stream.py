# SPDX-FileCopyrightText: © 2026 koderchina
# SPDX-License-Identifier: Apache-2.0
"""
audit_stream.py — throwaway diagnostic, NOT part of the regression.

Answers one question: do the streamed columns actually reach the datapath, or
do the normal tests pass for reasons unrelated to the data?

Run with:  make MODULE=audit_stream COCOTB_TEST_MODULES=audit_stream

Three independent lines of evidence:

  1. census      — count the bytes the DUT's own handshake accepts, and check
                   the count against the protocol's exact expected figure.
  2. unsaturated — the regression's vectors clamp ~90% of outputs to INT_MAX=7,
                   so a match there proves less than it looks. This drives a
                   geometry where every pooled value is a distinct small number
                   traceable to specific streamed pixels.
  3. sensitivity — zero one input column at a time and confirm the pooled map
                   changes. If a column never reaches the math, zeroing it is
                   invisible.
"""

import cocotb
from cocotb.triggers import RisingEdge, ReadOnly

from tt_testbench import (
    H_IN,
    IN_CHANNELS,
    K,
    OUT_BYTES,
    POOL_H,
    SETTLE,
    W_IN,
    expected_values,
    program,
    run_frame,
    start_dut,
    stream_and_collect,
)

# What the protocol says one frame must move across the pins.
STREAM_BYTES = W_IN * IN_CHANNELS * (H_IN // 2)


def bit(sig):
    """Read a 1-bit signal, tolerating x/z (cocotb raises on int() of those)."""
    try:
        return int(sig.value)
    except ValueError:
        return 0


class XferCounter:
    """Counts cycles where a handshake actually completed.

    Uses tb.v's in_xfer / out_xfer utility wires (valid && ready), sampled with
    the same SETTLE -> ReadOnly -> RisingEdge discipline the driver uses, so a
    transfer is counted exactly when the DUT commits to it.
    """

    def __init__(self, dut):
        self.dut = dut
        self.in_bytes = 0
        self.out_bytes = 0
        self.enabled = True

    async def run(self):
        while True:
            await SETTLE
            await ReadOnly()
            i = bit(self.dut.in_xfer)
            o = bit(self.dut.out_xfer)
            await RisingEdge(self.dut.clk)
            if self.enabled:
                self.in_bytes += i
                self.out_bytes += o


LOW, HIGH = 1, 3


def tap_kernel(kh=1, kw=1):
    """One unit tap on channel 0; channel 1 contributes nothing.

    Output = the channel-0 pixel at the tap's offset, so each pooled value is
    max() over a 2x2 patch of actual streamed pixels -- small, distinct, and
    individually traceable, unlike the saturated regression vectors.

    The tap POSITION selects which output column an input column lands in:
    centre (1,1) maps input col c -> pooled col c//2, while left (1,0) shifts
    it one to the right -> pooled col (c+1)//2. Two probes with different
    offsets give every input column a unique signature (see below).
    """
    kernels = [[[0] * K for _ in range(K)] for _ in range(IN_CHANNELS)]
    kernels[0][kh][kw] = 1
    return kernels


def single_tap_kernels():
    return tap_kernel(1, 1)


def flat_input(value=LOW):
    """Uniform low background, so raising one column makes it the pool maximum
    and therefore always observable."""
    return [[[value] * W_IN for _ in range(H_IN)] for _ in range(IN_CHANNELS)]


def ramp_input():
    """Distinct small values per position, so a dropped or misordered column
    shows up as a wrong number rather than another 7."""
    return [[[(r * W_IN + c + ic * 5) % 4 for c in range(W_IN)]
             for r in range(H_IN)] for ic in range(IN_CHANNELS)]


@cocotb.test()
async def test_audit_byte_census(dut):
    """The DUT's own handshake must accept exactly STREAM_BYTES input bytes and
    emit exactly OUT_BYTES output bytes for one frame."""
    pins, mon = await start_dut(dut)
    ctr = XferCounter(dut)
    cocotb.start_soon(ctr.run())

    kernels = single_tap_kernels()
    inputs = ramp_input()

    await program(dut, pins, kernels, 0)
    prog_bytes = ctr.in_bytes
    dut._log.info("PROGRAM accepted %d bytes", prog_bytes)

    ctr.in_bytes = 0
    await stream_and_collect(dut, pins, inputs, mon)

    dut._log.info("STREAM accepted %d input bytes, emitted %d output bytes",
                  ctr.in_bytes, ctr.out_bytes)

    assert ctr.in_bytes == STREAM_BYTES, (
        f"DUT accepted {ctr.in_bytes} stream bytes, protocol requires "
        f"{STREAM_BYTES} (= W_IN {W_IN} * IN_CHANNELS {IN_CHANNELS} * H_IN/2 "
        f"{H_IN // 2}). A mismatch means columns are not reaching the datapath."
    )
    assert ctr.out_bytes == OUT_BYTES, (
        f"DUT emitted {ctr.out_bytes} pooled bytes, expected {OUT_BYTES}"
    )
    dut._log.info("PASS — census exact: %d in / %d out",
                  ctr.in_bytes, ctr.out_bytes)


@cocotb.test()
async def test_audit_unsaturated_vectors(dut):
    """Same check the regression makes, but on a geometry that does NOT clamp.

    Fails loudly if the pooled map is all-7 (the regression's blind spot) or
    all one value (which any constant-output design would satisfy).
    """
    pins, mon = await start_dut(dut)

    kernels = single_tap_kernels()
    inputs = ramp_input()

    got = await run_frame(dut, pins, kernels, 0, inputs, mon)
    exp = expected_values(inputs, kernels, 0)

    dut._log.info("pooled map      : %s", got)
    dut._log.info("distinct values : %s", sorted(set(got)))

    assert got == exp
    assert len(set(got)) > 1, (
        f"every pooled value identical ({got[0]}) — this vector cannot "
        f"distinguish a working datapath from a constant"
    )
    assert set(got) != {7}, "still saturated; vector proves nothing"
    dut._log.info("PASS — %d distinct unsaturated values across %d outputs",
                  len(set(got)), len(got))


@cocotb.test()
async def test_audit_column_sensitivity(dut):
    """Raise one input column above the background; the pooled map must react,
    and react in the RIGHT pooled column.

    Do NOT perturb by zeroing. Max-pooling discards the smaller member of each
    2x2 block, so zeroing a column that was already being masked is invisible
    by construction -- that produces a false "column never reached the
    datapath" for every even column. Raising the column to the block maximum
    instead forces the pool to take it, so silence is genuinely diagnostic.

    Two probes, because one is not enough: input columns 2n and 2n+1 share a
    pooled column, so the centre tap alone cannot tell them apart. The left tap
    shifts each column one position right, and the pair of reactions gives all
    W_IN columns a unique signature. Under the left tap the last column shifts
    off the edge of the output map and correctly reacts nowhere -- that is the
    expected result, and it is what distinguishes it from its pair partner.
    """
    pins, mon = await start_dut(dut)

    signatures = {c: [] for c in range(W_IN)}

    for label, (kh, kw) in [("centre", (1, 1)), ("left", (1, 0))]:
        kernels = tap_kernel(kh, kw)
        baseline = await run_frame(dut, pins, kernels, 0, flat_input(), mon)
        dut._log.info("%s tap (%d,%d) baseline: %s", label, kh, kw, baseline)

        for col in range(W_IN):
            perturbed = flat_input()
            for r in range(H_IN):
                perturbed[0][r][col] = HIGH

            got = await run_frame(dut, pins, kernels, 0, perturbed, mon,
                                  do_program=False)
            reacted = tuple(sorted({
                i // POOL_H
                for i, (a, b) in enumerate(zip(baseline, got)) if a != b
            }))
            signatures[col].append(reacted)
            dut._log.info("  %s tap, col %d raised to %d -> reacted pooled "
                          "col(s) %s", label, col, HIGH, reacted or "() none")

    # Under the centre tap every column must land somewhere.
    inert = [c for c in range(W_IN) if not signatures[c][0]]
    assert not inert, (
        f"input column(s) {inert} produced no output change under the centre "
        f"tap — they are not reaching the datapath"
    )

    # And the two probes together must separate all of them.
    seen = {}
    collisions = []
    for col, sig in signatures.items():
        key = tuple(sig)
        if key in seen:
            collisions.append((seen[key], col, key))
        seen[key] = col

    dut._log.info("signatures: %s", {c: s for c, s in signatures.items()})
    assert not collisions, (
        f"input columns indistinguishable at the output: {collisions} — "
        f"cannot prove each column lands in its own position"
    )
    dut._log.info("PASS — all %d input columns individually observable and "
                  "uniquely located", W_IN)
