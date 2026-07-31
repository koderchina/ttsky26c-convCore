// =============================================================================
//  convLayer.v  —  pad -> conv -> bias -> ReLU -> MaxPool -> saturate
// =============================================================================
//
//  ONE physical conv instance and ONE of everything downstream of it.
//  Multi-input-channel convolution is serialized through the conv instance by
//  pad.v's ic replay and accumulated inside conv.v's circular buffer, so
//  conv.v's output is already the fully accumulated result -- there is no
//  per-channel conv array and no adder tree here.
//
//  There is likewise no output-channel array: an output channel is an
//  independent full convolution, so the chip computes exactly one per pass and
//  the host re-programs the weights and re-streams the image for the next one.
//  A single ReLU/MaxPool/saturate chain therefore suffices, and the maxpool's
//  own valid pulse IS the output pulse -- no hold registers, no gather stage.
//
//  Geometry is RUNTIME. MAX_* parameters bound widths only.
//
// =============================================================================

`timescale 1ns/1ps

module convLayer #(
    parameter DATA_WIDTH       = 4,
    parameter KERNEL_SIZE      = 3,
    parameter MAX_H_IN         = 32,
    parameter MAX_W_IN         = 1024,
    parameter MAX_IN_CHANNELS  = 4,
    parameter MAX_PAD          = 1,

    // ---- Derived — do not override ------------------------------------------
    localparam MAX_PADDED_HEIGHT = MAX_H_IN + 2*MAX_PAD,
    localparam MAX_PADDED_WIDTH  = MAX_W_IN + 2*MAX_PAD,
    localparam MAX_OUTPUT_HEIGHT = MAX_PADDED_HEIGHT - KERNEL_SIZE + 1,
    localparam MAX_OUTPUT_WIDTH  = MAX_PADDED_WIDTH  - KERNEL_SIZE + 1,
    localparam MAX_POOL_HEIGHT   = MAX_OUTPUT_HEIGHT / 2,
    localparam MAX_POOL_WIDTH    = MAX_OUTPUT_WIDTH  / 2,

    // Must match conv.v's derivation exactly.
    localparam ACC_WIDTH = 2*DATA_WIDTH - 1
                           + $clog2(KERNEL_SIZE*KERNEL_SIZE*MAX_IN_CHANNELS),
    // Only the bias add widens now -- the IN_CHANNELS adder tree is gone.
    localparam SUM_WIDTH = ACC_WIDTH + 1,

    localparam H_W        = $clog2(MAX_H_IN + 1),
    localparam W_W        = $clog2(MAX_W_IN + 1),
    localparam IC_W       = $clog2(MAX_IN_CHANNELS + 1),
    localparam PAD_W_W    = $clog2(MAX_PAD + 1),
    localparam IC_SEL_W   = (MAX_IN_CHANNELS <= 1) ? 1 : $clog2(MAX_IN_CHANNELS),
    localparam PADH_W     = $clog2(MAX_PADDED_HEIGHT + 1),
    localparam PADW_W     = $clog2(MAX_PADDED_WIDTH  + 1),
    localparam OUTH_W     = $clog2(MAX_OUTPUT_HEIGHT),
    localparam OUTW_W     = $clog2(MAX_OUTPUT_WIDTH),
    localparam POOL_ROW_BITS = (MAX_POOL_HEIGHT > 1) ? $clog2(MAX_POOL_HEIGHT) : 1,
    localparam POOL_COL_BITS = $clog2(MAX_POOL_WIDTH)
)(
    input  wire clk,
    input  wire rst_n,

    // ---- Runtime geometry ---------------------------------------------------
    input  wire [H_W-1:0]     i_h_in,
    input  wire [W_W-1:0]     i_w_in,
    input  wire [IC_W-1:0]    i_in_channels,
    input  wire [PAD_W_W-1:0] i_pad_w,
    input  wire [PAD_W_W-1:0] i_pad_h,

    // ---- Sample stream — one input channel's column at a time ---------------
    input  wire                         i_valid,
    input  wire                         start,
    input  wire signed [DATA_WIDTH-1:0] i_data,

    // Weights/bias from convCore's regfiles. Layout is the regfile's own: one
    // K*K*DATA_WIDTH slice per input channel at ic*K*K. pad.v produces that
    // index directly, so nothing is repacked here.
    input  wire [MAX_IN_CHANNELS*KERNEL_SIZE*KERNEL_SIZE*DATA_WIDTH-1:0] weights_in,
    input  wire [DATA_WIDTH-1:0]                                         biases_in,

    // Downstream readiness -- passthrough to pad's i_ds_ready.
    input  wire                                    i_ds_ready,

    output wire                                    o_valid,
    output wire signed [DATA_WIDTH-1:0]            o_data,
    output wire [POOL_ROW_BITS-1:0]                o_row,
    output wire [POOL_COL_BITS-1:0]                o_col,
    output wire                                    o_ready
);

// ---- Runtime padded geometry -------------------------------------------------
wire [PADW_W-1:0] padded_w = i_w_in + {i_pad_w, 1'b0};
wire [PADH_W-1:0] padded_h = i_h_in + {i_pad_h, 1'b0};

localparam signed [SUM_WIDTH-1:0] SAT_MAX =
    {{(SUM_WIDTH-DATA_WIDTH){1'b0}}, 1'b0, {(DATA_WIDTH-1){1'b1}}};
localparam signed [SUM_WIDTH-1:0] SAT_MIN =
    {{(SUM_WIDTH-DATA_WIDTH){1'b1}}, 1'b1, {(DATA_WIDTH-1){1'b0}}};

// =============================================================================
// Stage 0: Zero-padding + the input-channel replica sequencer
// =============================================================================
wire                       pad_o_valid;
wire                       pad_o_start;
wire signed [DATA_WIDTH-1:0] pad_o_data;
wire [IC_SEL_W-1:0]        pad_o_ic_sel;
wire                       pad_o_ic_last;

pad #(
    .DATA_WIDTH       (DATA_WIDTH),
    .MAX_H_IN         (MAX_H_IN),
    .MAX_W_IN         (MAX_W_IN),
    .MAX_IN_CHANNELS  (MAX_IN_CHANNELS),
    .MAX_PAD          (MAX_PAD)
) u_pad (
    .clk            (clk),
    .rst_n          (rst_n),
    .i_h_in         (i_h_in),
    .i_w_in         (i_w_in),
    .i_in_channels  (i_in_channels),
    .i_pad_w        (i_pad_w),
    .i_pad_h        (i_pad_h),
    .i_valid        (i_valid),
    .i_data         (i_data),
    .i_start        (start),
    .i_ds_ready     (i_ds_ready),
    .o_valid        (pad_o_valid),
    .o_data         (pad_o_data),
    .o_start        (pad_o_start),
    .o_ic_sel       (pad_o_ic_sel),
    .o_ic_last      (pad_o_ic_last),
    .o_ready        (o_ready)
);

// =============================================================================
// Stage 1: Convolution — ONE instance, accumulating all input channels
// =============================================================================
wire                        sum_valid;
wire signed [ACC_WIDTH-1:0] conv_data;
wire [OUTH_W-1:0]           sum_row;
wire [OUTW_W-1:0]           sum_col;

conv #(
    .DATA_WIDTH       (DATA_WIDTH),
    .KERNEL_SIZE      (KERNEL_SIZE),
    .MAX_H_IN         (MAX_H_IN),
    .MAX_W_IN         (MAX_W_IN),
    .MAX_IN_CHANNELS  (MAX_IN_CHANNELS),
    .MAX_PAD          (MAX_PAD)
) u_conv (
    .clk            (clk),
    .rst_n          (rst_n),
    .i_padded_w     (padded_w),
    .i_padded_h     (padded_h),
    .start          (pad_o_start),
    .i_valid        (pad_o_valid),
    .i_data         (pad_o_data),
    .weights_in     (weights_in),
    .ic_sel_in      (pad_o_ic_sel),
    .ic_last_in     (pad_o_ic_last),
    .o_valid        (sum_valid),
    .o_col          (sum_col),
    .o_row          (sum_row),
    .o_data         (conv_data),
    .done_flag      ()
);

// =============================================================================
// Stage 2: Bias addition — a single adder, one bias register
// =============================================================================
wire signed [SUM_WIDTH-1:0] sum_data =
    {{(SUM_WIDTH-ACC_WIDTH){conv_data[ACC_WIDTH-1]}}, conv_data};

wire signed [SUM_WIDTH-1:0] biased_data = sum_data +
    $signed({{(SUM_WIDTH-DATA_WIDTH){biases_in[DATA_WIDTH-1]}}, biases_in});

// =============================================================================
// Stage 3: ReLU
// =============================================================================
wire                        relu_valid;
wire signed [SUM_WIDTH-1:0] relu_data;
wire [OUTH_W-1:0]           relu_row;
wire [OUTW_W-1:0]           relu_col;

relu #(
    .DATA_WIDTH   (SUM_WIDTH),
    .INPUT_WIDTH  (MAX_OUTPUT_WIDTH),
    .INPUT_HEIGHT (MAX_OUTPUT_HEIGHT)
) u_relu (
    .i_valid (sum_valid),
    .i_data  (biased_data),
    .i_row   (sum_row),
    .i_col   (sum_col),
    .o_valid (relu_valid),
    .o_data  (relu_data),
    .o_row   (relu_row),
    .o_col   (relu_col)
);

// =============================================================================
// Stage 4: Max pool — uses only row/col parity, so nothing here depends on the
// runtime geometry.
// =============================================================================
wire                        pool_valid;
wire signed [SUM_WIDTH-1:0] pool_data;

maxpool #(
    .DATA_WIDTH   (SUM_WIDTH),
    .INPUT_WIDTH  (MAX_OUTPUT_WIDTH),
    .INPUT_HEIGHT (MAX_OUTPUT_HEIGHT),
    .POOL_WIDTH   (MAX_POOL_WIDTH),
    .POOL_HEIGHT  (MAX_POOL_HEIGHT)
) u_maxpool (
    .clk     (clk),
    .rst_n   (rst_n),
    .i_valid (relu_valid),
    .i_data  (relu_data),
    .i_row   (relu_row),
    .i_col   (relu_col),
    .o_valid (pool_valid),
    .o_data  (pool_data),
    .o_row   (o_row),
    .o_col   (o_col)
);

// =============================================================================
// Stage 5: Saturation — read live on the pool pulse. With a single channel in
// flight there is nothing to hold or re-gather: the pool's own pulse is the
// output pulse.
// =============================================================================
assign o_data =
    (pool_data > SAT_MAX) ? SAT_MAX[DATA_WIDTH-1:0] :
    (pool_data < SAT_MIN) ? SAT_MIN[DATA_WIDTH-1:0] :
     pool_data[DATA_WIDTH-1:0];

assign o_valid = pool_valid;

endmodule
