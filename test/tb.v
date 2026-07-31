`default_nettype none
`timescale 1ns / 1ps

/* This testbench just instantiates the module and makes some convenient wires
   that can be driven / tested by the cocotb test.py.
*/
module tb ();

  // Dump the signals to a FST file. You can view it with gtkwave or surfer.
  initial begin
    $dumpfile("tb.fst");
    $dumpvars(0, tb);
    #1;
  end

  // Wire up the inputs and outputs:
  reg clk;
  reg rst_n;
  reg ena;
  reg [7:0] ui_in;
  reg [7:0] uio_in;
  wire [7:0] uo_out;
  wire [7:0] uio_out;
  wire [7:0] uio_oe;
`ifdef GL_TEST
  wire VPWR = 1'b1;
  wire VGND = 1'b0;
`endif

  // ---- Utility wires --------------------------------------------------------
  // Named aliases for the packed handshake bits, so a waveform shows the
  // protocol instead of anonymous uio slices. Read-only: the cocotb driver
  // still owns ui_in / uio_in as whole bytes (see Pins in tt_testbench.py --
  // mode/in_valid and out_ready share uio_in, so one writer composes it all).

  // Host -> design, uio_in[2:0]
  wire       mode      = uio_in[0];   // 0 = PROGRAM, 1 = STREAM
  wire       in_valid  = uio_in[1];   // ui_in holds a byte this cycle
  wire       out_ready = uio_in[2];   // host can accept an output byte

  // Design -> host, uio_out[5:3]
  wire       in_ready  = uio_out[3];  // design can accept a byte on ui_in
  wire       out_valid = uio_out[4];  // uo_out holds a valid pooled byte
  wire       busy      = uio_out[5];  // mid-frame (STREAM / DRAIN)

  // Nibble views of the shared data byte. PROGRAM and STREAM both send two
  // 4-bit values per byte, low nibble first; output carries one pooled value
  // per byte in the low nibble, high nibble zero.
  wire [3:0] in_lo     = ui_in[3:0];
  wire [3:0] in_hi     = ui_in[7:4];
  wire [3:0] out_value = uo_out[3:0];

  // A byte only actually moves on an edge where both sides agree.
  wire       in_xfer   = in_valid  && in_ready;
  wire       out_xfer  = out_valid && out_ready;

  tt_um_koderchina_convCore user_project (

      // Include power ports for the Gate Level test:
`ifdef GL_TEST
      .VPWR(VPWR),
      .VGND(VGND),
`endif

      .ui_in  (ui_in),    // Dedicated inputs
      .uo_out (uo_out),   // Dedicated outputs
      .uio_in (uio_in),   // IOs: Input path
      .uio_out(uio_out),  // IOs: Output path
      .uio_oe (uio_oe),   // IOs: Enable path (active high: 0=input, 1=output)
      .ena    (ena),      // enable - goes high when design is selected
      .clk    (clk),      // clock
      .rst_n  (rst_n)     // not reset
  );

endmodule
