/*
 * Copyright (c) 2026 koderchina
 * SPDX-License-Identifier: Apache-2.0
 *
 * TinyTapeout wrapper for the programmable convolution accelerator.
 * The real design lives in top.v (and the modules below it); this file only
 * adapts it to the tt_um_* pin contract.
 *
 * PIN MAP (see top.v's header for the full protocol)
 *   ui_in[7:0]   data byte — TWO 4-bit values, low nibble first
 *   uio_in[0]    mode       0 = PROGRAM (write config/weight/bias), 1 = STREAM
 *   uio_in[1]    in_valid   ui_in holds a byte this cycle
 *   uio_in[2]    out_ready  host can accept an output byte this cycle
 *   uio_in[7:3]  unused     (inputs, ignored)
 *   uo_out[3:0]  pooled output value   uo_out[7:4] = 0
 *   uio_out[3]   in_ready   design can accept a byte on ui_in this cycle
 *   uio_out[4]   out_valid  uo_out holds a valid pooled byte this cycle
 *   uio_out[5]   busy       1 whenever mid-frame (STREAM / DRAIN)
 *   uio_out[7:6], uio_out[2:0]  unused, driven to 0
 */

`default_nettype none

module tt_um_koderchina_convCore (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (active high: 0=input, 1=output)
    input  wire       ena,      // always 1 when the design is powered, so you can ignore it
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);

  // The design already resets active-low on rst_n, exactly like the tt_um_*
  // contract, so no inverter / re-polarised reset is needed here: rst_n goes
  // straight through.
  //
  // ena is ignored. top.v has its own ena port that gates its outputs, so it is
  // tied high to make that gating a constant (synthesis folds it away) and the
  // real ena pin is absorbed by _unused below. That leaves uio_oe a STATIC
  // 8'b0011_1000: uio[5:3] are outputs (busy / out_valid / in_ready), all other
  // bidirectional pins are inputs.
  top #(
      .DATA_WIDTH      (4),     // nibble width; two per pin byte
      .KERNEL_SIZE     (3),
      .MAX_H_IN        (16),
      .MAX_W_IN        (1024),
      .MAX_IN_CHANNELS (4),
      .MAX_PAD         (1),
      .REG_ADDR_W      (9)      // must cover TOTAL_REGS
  ) u_top (
      .clk     (clk),
      .rst_n   (rst_n),
      .ena     (1'b1),

      .ui_in   (ui_in),
      .uo_out  (uo_out),
      .uio_in  (uio_in),
      .uio_out (uio_out),
      .uio_oe  (uio_oe)
  );

  // List all unused inputs to prevent warnings
  wire _unused = &{ena, 1'b0};

endmodule

`default_nettype wire
