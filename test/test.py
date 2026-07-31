# SPDX-FileCopyrightText: © 2026 koderchina
# SPDX-License-Identifier: Apache-2.0
"""
test.py — cocotb entry points for tt_um_koderchina_convCore
===========================================================

This is the module named by COCOTB_TEST_MODULES. It holds only the
@cocotb.test() functions; the pin-level protocol driver, the register-image
builder and the golden reference model all live in tt_testbench.py.

`dut` is tb.v (TOPLEVEL = tb), which mirrors the tt_um_* port names, so the
driver in tt_testbench.py addresses dut.ui_in / dut.uio_out / ... directly.

Run everything with `make`. For a single test the variable name follows the
installed cocotb: `make TESTCASE=test_top_slow_host` on 1.x, or
`make COCOTB_TESTCASE=test_top_slow_host` on >=2.0.
"""

import random

import cocotb

from tt_testbench import (
    H_IN,
    IN_CHANNELS,
    K,
    NUM_OUT_CHANNELS,
    W_IN,
    expected_values,
    random_input,
    random_kernels,
    run_frame,
    start_dut,
)


@cocotb.test()
async def test_top_zeros(dut):
    """Zero input and zero weights: every pooled value must be 0."""
    pins, mon = await start_dut(dut)

    kernels = [[[0] * K for _ in range(K)] for _ in range(IN_CHANNELS)]
    inputs  = [[[0] * W_IN for _ in range(H_IN)] for _ in range(IN_CHANNELS)]

    got = await run_frame(dut, pins, kernels, 0, inputs, mon)
    assert all(v == 0 for v in got)
    dut._log.info("PASS — %d all-zero pooled values", len(got))


@cocotb.test()
async def test_top_random_multichannel(dut):
    """IN_CHANNELS>1 with a non-zero bias — the ic replica sweep end to end."""
    pins, mon = await start_dut(dut)

    kernels = random_kernels(2026)
    bias    = random.randint(-2, 2)
    inputs  = random_input()

    got = await run_frame(dut, pins, kernels, bias, inputs, mon)
    dut._log.info("PASS — %d pooled values", len(got))


@cocotb.test()
async def test_top_back_to_back(dut):
    """Two frames without an intervening reset or reprogram: entering STREAM
    pulses the datapath-only reset, while the register files sit on their own
    reg_rst_n and keep the programmed weights."""
    pins, mon = await start_dut(dut)

    kernels = random_kernels(77)
    input_a = random_input()
    input_b = random_input()

    await run_frame(dut, pins, kernels, 0, input_a, mon)
    dut._log.info("Frame A PASS")

    # No reset, no reprogram — stream straight into frame B.
    await run_frame(dut, pins, kernels, 0, input_b, mon, do_program=False)
    dut._log.info("Frame B PASS")


@cocotb.test()
async def test_top_multi_output_channel_passes(dut):
    """The host protocol: the SAME input image convolved into NUM_OUT_CHANNELS
    independent feature maps, one pass each.

    Per channel the host reprograms the weights/bias (returning to PROGRAM via
    the existing DRAIN->PROG path, no new states) and re-streams the whole
    image. Stacking the results is off-chip work, so this test does it in
    Python and checks the stack against the reference. Distinct kernels per
    channel are what catches a datapath that leaked state between passes --
    a stale accumulator or an unreset weight bank would make pass N look like
    pass N-1.
    """
    pins, mon = await start_dut(dut)

    inputs   = random_input()
    kernels  = [random_kernels(1000 + oc) for oc in range(NUM_OUT_CHANNELS)]
    biases   = [oc - 1 for oc in range(NUM_OUT_CHANNELS)]   # -1, 0, +1

    stacked = []
    for oc in range(NUM_OUT_CHANNELS):
        got = await run_frame(dut, pins, kernels[oc], biases[oc], inputs, mon)
        stacked.append(got)
        dut._log.info("Output channel %d PASS (%d values)", oc, len(got))

    expected = [expected_values(inputs, kernels[oc], biases[oc])
                for oc in range(NUM_OUT_CHANNELS)]
    assert stacked == expected, "stacked multi-channel result mismatch"

    # Distinct kernels must give distinct maps, or the test proves nothing
    # about state leaking between passes.
    assert len(set(tuple(m) for m in stacked)) > 1, (
        "all output channels produced identical maps — "
        "the per-pass reprogram is not taking effect"
    )
    dut._log.info("PASS — %d output channels stacked off-chip",
                  NUM_OUT_CHANNELS)


@cocotb.test()
async def test_top_slow_host(dut):
    """A host that cannot keep up with the emit burst.

    conv.v's emit side is free-running, so during an odd output column's burst
    maxpool delivers a pooled value every 2 cycles no matter what the host is
    doing. This pattern raises out_ready roughly one cycle in seven -- far
    slower than production -- so every burst has to sit in the output FIFO
    while the host dribbles it out. Values would be silently lost without it.

    The result must be bit-identical to the same frame drained at full speed,
    which test_top_random_multichannel already checks against the reference.
    """
    pins, mon = await start_dut(dut, ready_pattern=[1, 0, 0, 0, 0, 0, 0])

    kernels = random_kernels(2026)
    bias    = random.randint(-2, 2)
    inputs  = random_input()

    got = await run_frame(dut, pins, kernels, bias, inputs, mon)
    assert got == expected_values(inputs, kernels, bias)
    dut._log.info("PASS — %d values drained through the FIFO by a slow host",
                  len(got))


@cocotb.test()
async def test_top_slow_host_multi_pass(dut):
    """The slow host again, across the per-output-channel reprogram loop --
    the FIFO must come up empty for each new pass, not carrying values over
    from the previous one."""
    pins, mon = await start_dut(dut, ready_pattern=[1, 1, 0, 0, 0])

    inputs  = random_input()
    kernels = [random_kernels(1000 + oc) for oc in range(NUM_OUT_CHANNELS)]
    biases  = [oc - 1 for oc in range(NUM_OUT_CHANNELS)]

    for oc in range(NUM_OUT_CHANNELS):
        got = await run_frame(dut, pins, kernels[oc], biases[oc], inputs, mon)
        assert got == expected_values(inputs, kernels[oc], biases[oc]), (
            f"output channel {oc} mismatch under a stalling host"
        )
    dut._log.info("PASS — %d slow-host passes, no cross-pass FIFO carryover",
                  NUM_OUT_CHANNELS)
