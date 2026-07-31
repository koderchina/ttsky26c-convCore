// =============================================================================
//  config_regfile.v  —  runtime geometry/configuration registers
// =============================================================================
//
//  Holds the five values that used to be compile-time Verilog parameters on
//  pad/conv/convLayer. They are now written at runtime over the same
//  PROGRAM-phase byte-serial path the weights use, so one hardened core can
//  run any geometry inside the MAX_* envelope it was built for.
//
//  There is deliberately NO OUT_CHANNELS field: an output channel is an
//  independent full convolution, so the chip computes exactly one of them per
//  pass and the host re-programs + re-streams for the next. Nothing on-chip
//  needs to know how many channels the host intends to collect.
//
//  Address space is measured in NIBBLES (4-bit registers), matching the rest
//  of the register map -- top.v splits every incoming 8-bit pin byte into two
//  consecutive nibble writes, so DATA_WIDTH here is the nibble width.
//
//  Field layout — 5 fields, NIB_PER_FIELD nibbles each, little-endian
//  (nibble j of a field carries that field's bits [4*j+3 : 4*j]):
//
//    nibble  0.. 3   H_IN            nibble 12..15   PAD_W
//    nibble  4.. 7   W_IN            nibble 16..19   PAD_H
//    nibble  8..11   IN_CHANNELS
//
//  Every field spends the same NIB_PER_FIELD nibbles on the wire so the host
//  format is uniform and the address decode stays a bit-slice (NIB_PER_FIELD
//  is a power of two -- no divider). Storage, however, is only as wide as the
//  MAX_* envelope requires: nibbles above NIB_STORED are accepted and
//  discarded rather than stored.
//
//  Reads are combinational, so a value is visible to the datapath the cycle
//  after its last stored nibble lands.
//
// =============================================================================

`timescale 1ns / 1ps

module config_regfile #(
    parameter DATA_WIDTH       = 4,     // nibble width — one register per address
    parameter REG_ADDR_W       = 9,

    // Compile-time envelope. Bounds the stored register widths only; the
    // on-the-wire format is independent of it.
    parameter MAX_H_IN         = 32,
    parameter MAX_W_IN         = 1024,
    parameter MAX_IN_CHANNELS  = 4,
    parameter MAX_PAD          = 1,

    // ---- Derived widths — do not override -----------------------------------
    // Each field holds a COUNT, not a max index, so the +1 is deliberate:
    // H_IN may legitimately equal MAX_H_IN.
    localparam H_W     = $clog2(MAX_H_IN         + 1),
    localparam W_W     = $clog2(MAX_W_IN         + 1),
    localparam IC_W    = $clog2(MAX_IN_CHANNELS  + 1),
    localparam PAD_W_W = $clog2(MAX_PAD          + 1)
)(
    input  wire                  clk,
    input  wire                  rst_n,

    // Register write port — single-cycle, no handshake (same shape as
    // weight_regfile.v). wr_addr is the GLOBAL nibble address; this module
    // decodes its own range, so the caller can fan the bus out unconditionally.
    input  wire                  wr_en,
    input  wire [REG_ADDR_W-1:0] wr_addr,
    input  wire [DATA_WIDTH-1:0] wr_data,

    // Combinational config outputs
    output wire [H_W-1:0]        o_h_in,
    output wire [W_W-1:0]        o_w_in,
    output wire [IC_W-1:0]       o_in_channels,
    output wire [PAD_W_W-1:0]    o_pad_w,
    output wire [PAD_W_W-1:0]    o_pad_h
);

localparam N_FIELDS      = 5;
localparam NIB_PER_FIELD = 4;                        // power of 2 -> decode is a bit-slice
localparam CFG_NIBBLES   = N_FIELDS * NIB_PER_FIELD; // = 20

// Storage width, common to all five fields and rounded up to a whole number of
// nibbles so every stored nibble is a clean, in-range part-select. The widest
// field sets it (W_IN in any sane envelope); narrower fields carry dangling
// upper bits that synthesis trims, since nothing downstream reads them.
localparam CFG_BITS_A  = (H_W  > W_W)     ? H_W  : W_W;
localparam CFG_BITS_B  = (IC_W > PAD_W_W) ? IC_W : PAD_W_W;
localparam CFG_BITS    = (CFG_BITS_A > CFG_BITS_B) ? CFG_BITS_A : CFG_BITS_B;

localparam NIB_CEIL    = (CFG_BITS + DATA_WIDTH - 1) / DATA_WIDTH;
// Never keep more nibbles than the wire format carries.
localparam NIB_STORED  = (NIB_CEIL > NIB_PER_FIELD) ? NIB_PER_FIELD : NIB_CEIL;
localparam CFG_FIELD_W = NIB_STORED * DATA_WIDTH;

// ---- Storage ----------------------------------------------------------------
reg [CFG_FIELD_W-1:0] r_cfg [0:N_FIELDS-1];

wire                  wr_is_cfg = wr_en && (wr_addr < CFG_NIBBLES);
wire [REG_ADDR_W-1:0] field_idx = wr_addr / NIB_PER_FIELD;   // power-of-2 -> bit slice
wire [REG_ADDR_W-1:0] nib_idx   = wr_addr % NIB_PER_FIELD;

integer f;
always @(posedge clk) begin
    if (!rst_n) begin
        for (f = 0; f < N_FIELDS; f = f + 1) r_cfg[f] <= {CFG_FIELD_W{1'b0}};
    end else if (wr_is_cfg && (nib_idx < NIB_STORED)) begin
        // Nibbles at or above NIB_STORED are part of the uniform wire format
        // but carry no information this envelope can represent -- accept and
        // drop them rather than doing an out-of-range part-select.
        r_cfg[field_idx][nib_idx*DATA_WIDTH +: DATA_WIDTH] <= wr_data;
    end
end

// ---- Outputs ----------------------------------------------------------------
assign o_h_in         = r_cfg[0][H_W-1:0];
assign o_w_in         = r_cfg[1][W_W-1:0];
assign o_in_channels  = r_cfg[2][IC_W-1:0];
assign o_pad_w        = r_cfg[3][PAD_W_W-1:0];
assign o_pad_h        = r_cfg[4][PAD_W_W-1:0];

endmodule
