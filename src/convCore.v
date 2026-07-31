`timescale 1ns/1ps

// =============================================================================
//  convCore.v  —  the programmable convolution accelerator core
// =============================================================================
//
//  One Conv -> ReLU -> MaxPool stage plus the register files that configure it.
//  (It used to stack three such stages and a Global Average Pooling tail under
//  the name convBlock; the cascade, the GAP and the old name are all gone.)
//
//  The core computes ONE output channel per pass. An output channel is an
//  independent full convolution over all input channels, so nothing on-chip
//  needs the total channel count: the host programs one channel's weights and
//  bias, streams the image, collects that channel's pooled feature map, and
//  repeats for the next channel, stacking the maps off-chip.
//
//  Unified register map — addresses are NIBBLE (4-bit register) addresses,
//  written in order starting at 0. Every boundary is compile-time, derived
//  from the MAX_* envelope rather than the runtime channel count, so the
//  write decode needs no multipliers; the host simply writes a fixed-size
//  image and skips over unused channel slots.
//
//    nibble  0.. 19   config   H_IN, W_IN, IN_CHANNELS, PAD_W, PAD_H
//                              — 4 nibbles each, little-endian
//    nibble 20..      weights  addr = CFG_NIBBLES + ic*K*K + kw*K + tap
//    nibble ..        bias     addr = CFG_NIBBLES + MAX_IN_CHANNELS*K*K
//                              (a single register — one output channel)
//
//  The configuration values are re-exported so the TinyTapeout wrapper can
//  size its own stream counters and pooled-output bookkeeping from the same
//  registers the datapath uses.
//
// =============================================================================

module convCore #(
    parameter DATA_WIDTH       = 4,
    parameter KERNEL_SIZE      = 3,
    parameter MAX_H_IN         = 32,
    parameter MAX_W_IN         = 1024,
    parameter MAX_IN_CHANNELS  = 4,
    parameter MAX_PAD          = 1,
    parameter REG_ADDR_W       = 9,     // must cover TOTAL_REGS below

    // ---- Derived — do not override ------------------------------------------
    localparam CFG_NIBBLES       = 20,  // 5 fields x 4 nibbles
    localparam N_WEIGHT_REGS     = MAX_IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE,
    localparam TOTAL_REGS        = CFG_NIBBLES + N_WEIGHT_REGS + 1,

    localparam MAX_PADDED_HEIGHT = MAX_H_IN + 2*MAX_PAD,
    localparam MAX_PADDED_WIDTH  = MAX_W_IN + 2*MAX_PAD,
    localparam MAX_OUTPUT_HEIGHT = MAX_PADDED_HEIGHT - KERNEL_SIZE + 1,
    localparam MAX_OUTPUT_WIDTH  = MAX_PADDED_WIDTH  - KERNEL_SIZE + 1,
    localparam MAX_POOL_HEIGHT   = MAX_OUTPUT_HEIGHT / 2,
    localparam MAX_POOL_WIDTH    = MAX_OUTPUT_WIDTH  / 2,

    localparam H_W        = $clog2(MAX_H_IN + 1),
    localparam W_W        = $clog2(MAX_W_IN + 1),
    localparam IC_W       = $clog2(MAX_IN_CHANNELS + 1),
    localparam PAD_W_W    = $clog2(MAX_PAD + 1),
    localparam POOL_ROW_BITS = (MAX_POOL_HEIGHT > 1) ? $clog2(MAX_POOL_HEIGHT) : 1,
    localparam POOL_COL_BITS = $clog2(MAX_POOL_WIDTH)
)(
    input  wire clk,
    input  wire rst_n,      // datapath reset (pad, conv, pool)
    input  wire reg_rst_n,  // register-file reset (system reset only)

    // -------------------------------------------------------------------------
    // Streaming input — one input channel's column at a time
    // -------------------------------------------------------------------------
    input  wire                         i_valid,
    input  wire                         start,
    input  wire signed [DATA_WIDTH-1:0] i_data,
    output wire                         o_ready,

    // -------------------------------------------------------------------------
    // Unified config/weight/bias register file — single-cycle write port
    // -------------------------------------------------------------------------
    input  wire                         reg_wr_en,
    input  wire [REG_ADDR_W-1:0]        reg_wr_addr,
    input  wire [DATA_WIDTH-1:0]        reg_wr_data,

    // -------------------------------------------------------------------------
    // Pooled feature-map output — one value per pooled position
    // -------------------------------------------------------------------------
    input  wire                         i_ds_ready,
    output wire                         o_valid,
    output wire signed [DATA_WIDTH-1:0] o_data,
    output wire [POOL_ROW_BITS-1:0]     o_row,
    output wire [POOL_COL_BITS-1:0]     o_col,

    // -------------------------------------------------------------------------
    // Re-exported configuration (top.v drives its stream counters from these)
    // -------------------------------------------------------------------------
    output wire [H_W-1:0]     cfg_h_in,
    output wire [W_W-1:0]     cfg_w_in,
    output wire [IC_W-1:0]    cfg_in_channels,
    output wire [PAD_W_W-1:0] cfg_pad_w,
    output wire [PAD_W_W-1:0] cfg_pad_h
);

// =============================================================================
// Register-file address routing
//
// config_regfile range-checks its own window internally, so it can take the
// raw address. weight_regfile is addressed from its own base, so the config
// region's size is subtracted and its write enable gated.
// =============================================================================
wire                    wr_is_wb   = reg_wr_en && (reg_wr_addr >= CFG_NIBBLES);
wire [REG_ADDR_W-1:0]   wb_wr_addr = reg_wr_addr - CFG_NIBBLES;

config_regfile #(
    .DATA_WIDTH       (DATA_WIDTH),
    .REG_ADDR_W       (REG_ADDR_W),
    .MAX_H_IN         (MAX_H_IN),
    .MAX_W_IN         (MAX_W_IN),
    .MAX_IN_CHANNELS  (MAX_IN_CHANNELS),
    .MAX_PAD          (MAX_PAD)
) u_config (
    .clk            (clk),
    .rst_n          (reg_rst_n),
    .wr_en          (reg_wr_en),
    .wr_addr        (reg_wr_addr),
    .wr_data        (reg_wr_data),
    .o_h_in         (cfg_h_in),
    .o_w_in         (cfg_w_in),
    .o_in_channels  (cfg_in_channels),
    .o_pad_w        (cfg_pad_w),
    .o_pad_h        (cfg_pad_h)
);

wire [MAX_IN_CHANNELS*KERNEL_SIZE*KERNEL_SIZE*DATA_WIDTH-1:0] w_weights;
wire [DATA_WIDTH-1:0]                                         w_biases;

weight_regfile #(
    .DATA_WIDTH       (DATA_WIDTH),
    .KERNEL_SIZE      (KERNEL_SIZE),
    .MAX_IN_CHANNELS  (MAX_IN_CHANNELS),
    .REG_ADDR_W       (REG_ADDR_W)
) u_regfile (
    .clk         (clk),
    .rst_n       (reg_rst_n),
    .wr_en       (wr_is_wb),
    .wr_addr     (wb_wr_addr),
    .wr_data     (reg_wr_data),
    .weights_out (w_weights),
    .biases_out  (w_biases)
);

// =============================================================================
// The single accelerator stage
// =============================================================================
convLayer #(
    .DATA_WIDTH       (DATA_WIDTH),
    .KERNEL_SIZE      (KERNEL_SIZE),
    .MAX_H_IN         (MAX_H_IN),
    .MAX_W_IN         (MAX_W_IN),
    .MAX_IN_CHANNELS  (MAX_IN_CHANNELS),
    .MAX_PAD          (MAX_PAD)
) u_layer (
    .clk            (clk),
    .rst_n          (rst_n),
    .i_h_in         (cfg_h_in),
    .i_w_in         (cfg_w_in),
    .i_in_channels  (cfg_in_channels),
    .i_pad_w        (cfg_pad_w),
    .i_pad_h        (cfg_pad_h),
    .i_valid        (i_valid),
    .start          (start),
    .i_data         (i_data),
    .weights_in     (w_weights),
    .biases_in      (w_biases),
    // Backpressure from top.v's output byte-serializer. This is the same
    // handshake cascaded layers used to drive each other with; the serializer
    // has simply taken the downstream layer's place.
    .i_ds_ready     (i_ds_ready),
    .o_valid        (o_valid),
    .o_data         (o_data),
    .o_row          (o_row),
    .o_col          (o_col),
    .o_ready        (o_ready)
);

endmodule
