// =============================================================================
//  pad.v  —  zero-padding + the input-channel replica sequencer
// =============================================================================
//
//  Buffers ONE column of ONE input channel and replays it against every input
//  channel's weight slice, so a single physical conv/PE pipeline can compute a
//  full multi-input-channel convolution serially.
//
//  Replay nesting — column -> ic -> row:
//
//    for col in 0..PADDED_WIDTH-1:
//        for ic in 0..IN_CHANNELS-1:
//            FILL   H_IN fresh samples for this (col, ic)      # real cols only
//            EMIT   for row in 0..PADDED_HEIGHT-1:
//                       present the sample against ic's weight slice
//        -> advance the real column
//
//  There is NO output-channel axis. An output channel is an independent full
//  convolution, so the chip computes exactly one per pass and the host
//  re-programs the weights and re-streams the image for the next one. ic is
//  the only replica axis, which is what lets conv.v keep its small circular
//  accumulator: for a fixed column, every ic pass accumulates into the same
//  (row, slot) cell -- exactly "sum the contributions of all input channels".
//
//  LEFT_PAD / RIGHT_PAD zero-columns get the SAME ic replication (no FILL --
//  o_data stays 0) purely so conv.v's per-column counter advances uniformly
//  regardless of column type. Numerically harmless: zero contributes zero
//  under any channel's weights.
//
//  Outputs driving conv.v:
//    o_ic_sel    which input channel's weight slice this beat multiplies by
//    o_ic_last   this beat belongs to the final input channel -- conv.v uses
//                it (delayed) to fire "a real column has fully landed" once
//                per column instead of once per channel
//
//  Downstream backpressure (i_ds_ready): the position only advances on cycles
//  where i_ds_ready is high. Otherwise it freezes and o_valid drops (one cycle
//  later, tracked by r_stalled, since the outputs are registered), so conv.v
//  never re-accumulates a stalled beat.
//
//  Geometry (H_IN/W_IN/IN_CHANNELS/PAD_W/PAD_H) is RUNTIME; the MAX_*
//  parameters only bound register and array widths.
//
// =============================================================================

`timescale 1ns/1ps

module pad #(
    parameter DATA_WIDTH       = 4,
    parameter MAX_H_IN         = 32,
    parameter MAX_W_IN         = 1024,
    parameter MAX_IN_CHANNELS  = 4,
    parameter MAX_PAD          = 1,

    // ---- Derived — do not override ------------------------------------------
    localparam MAX_PADDED_HEIGHT = MAX_H_IN + 2*MAX_PAD,
    localparam MAX_PADDED_WIDTH  = MAX_W_IN + 2*MAX_PAD,

    localparam H_W        = $clog2(MAX_H_IN + 1),
    localparam W_W        = $clog2(MAX_W_IN + 1),
    localparam IC_W       = $clog2(MAX_IN_CHANNELS + 1),
    localparam PAD_W_W    = $clog2(MAX_PAD + 1),

    // Index (not count) width for the replica selector
    localparam IC_SEL_W   = (MAX_IN_CHANNELS  <= 1) ? 1 : $clog2(MAX_IN_CHANNELS),

    localparam ROW_W      = $clog2(MAX_PADDED_HEIGHT + 1),
    localparam PADCOL_W   = $clog2(MAX_PAD + 2)
)(
    input  wire                       clk,
    input  wire                       rst_n,

    // ---- Runtime geometry ---------------------------------------------------
    input  wire [H_W-1:0]             i_h_in,
    input  wire [W_W-1:0]             i_w_in,
    input  wire [IC_W-1:0]            i_in_channels,
    input  wire [PAD_W_W-1:0]         i_pad_w,
    input  wire [PAD_W_W-1:0]         i_pad_h,

    // ---- Upstream sample stream (ONE channel's column at a time) ------------
    input  wire                       i_valid,
    input  wire signed [DATA_WIDTH-1:0] i_data,
    input  wire                       i_start,

    // ---- Downstream readiness ----------------------------------------------
    input  wire                       i_ds_ready,

    // ---- Replayed output stream --------------------------------------------
    output reg                        o_valid,
    output reg  signed [DATA_WIDTH-1:0] o_data,
    output reg                        o_start,
    output reg  [IC_SEL_W-1:0]        o_ic_sel,
    output reg                        o_ic_last,
    output reg                        o_ready
);

// ---- Runtime padded geometry -----------------------------------------------
wire [ROW_W-1:0] padded_h = i_h_in + {i_pad_h, 1'b0};   // h_in + 2*pad_h

localparam IDLE      = 3'd0;
localparam LEFT_PAD  = 3'd1;
localparam FILL      = 3'd2;
localparam EMIT      = 3'd3;
localparam RIGHT_PAD = 3'd4;
localparam DONE      = 3'd5;

reg [2:0] r_state, r_next_state;

// One column of ONE input channel -- the whole point of the ic serialization.
reg signed [DATA_WIDTH-1:0] r_col_buf [0:MAX_H_IN-1];

reg [H_W-1:0]      r_fill_row;   // 0..h_in-1        while filling
reg [ROW_W-1:0]    r_out_row;    // 0..padded_h-1    while emitting
reg [PADCOL_W-1:0] r_out_col;    // 0..pad_w-1       within a pad run
reg [W_W-1:0]      r_real_col;   // 0..w_in          real columns emitted

// ---- Replica sequencer -------------------------------------------------------
reg [IC_SEL_W-1:0] r_ic_sel;

// The outputs are registered, so a beat presented while downstream was stalled
// must be squashed one cycle LATER -- hence a flag rather than a live
// !i_ds_ready term.
reg r_stalled;

wire ic_terminal = (i_in_channels <= 1) ? 1'b1 : (r_ic_sel == i_in_channels - 1);

// End of one full row pass for the current ic.
wire sweep_done  = (r_out_row == padded_h - 1) && i_ds_ready;
// End of the whole ic x row replay for the current column position.
wire col_done    = sweep_done && ic_terminal;

wire last_real_col = (r_real_col == i_w_in - 1);
wire last_pad_col  = (r_out_col  == i_pad_w - 1);

wire in_replay = (r_state == LEFT_PAD) || (r_state == EMIT) || (r_state == RIGHT_PAD);

// Which buffered row this emit beat is presenting (only meaningful inside the
// real-data band of the padded column). Outside that band r_out_row - i_pad_h
// underflows (r_out_row < i_pad_h during top padding); gate it to 0 there so
// synthesis never wires up a read of the unwritten tail of r_col_buf.
wire           in_band  = (r_out_row >= i_pad_h) && (r_out_row < i_pad_h + i_h_in);
wire [H_W-1:0] buf_idx  = in_band ? (r_out_row - i_pad_h) : {H_W{1'b0}};

// ---- Stall tracking ----------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n)          r_stalled <= 1'b0;
    else if (in_replay)  r_stalled <= !i_ds_ready;
    else                 r_stalled <= 1'b0;
end

// ---- ic sequencer ------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        r_ic_sel <= {IC_SEL_W{1'b0}};
    end else if (r_state == IDLE || r_state == DONE) begin
        r_ic_sel <= {IC_SEL_W{1'b0}};
    end else if (in_replay && sweep_done) begin
        if (ic_terminal) r_ic_sel <= {IC_SEL_W{1'b0}};
        else             r_ic_sel <= r_ic_sel + 1'b1;
    end
end

// ---- FSM --------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) r_state <= IDLE;
    else        r_state <= r_next_state;
end

always @(*) begin
    r_next_state = r_state;
    case (r_state)
        IDLE:      r_next_state = (i_pad_w != 0) ? LEFT_PAD : FILL;

        // Every pad column is replayed for all ic, exactly like a real one.
        LEFT_PAD:  if (col_done && last_pad_col) r_next_state = FILL;

        FILL:      if (r_fill_row == i_h_in - 1 && i_valid) r_next_state = EMIT;

        // Not the last ic  -> fetch the next channel's column for the SAME
        //                     real column position.
        // Last ic          -> the column is complete; move on.
        EMIT:      if (sweep_done)
                       r_next_state = (ic_terminal && last_real_col)
                                      ? ((i_pad_w != 0) ? RIGHT_PAD : DONE)
                                      : FILL;

        RIGHT_PAD: if (col_done && last_pad_col) r_next_state = DONE;

        DONE:      r_next_state = DONE;

        default:   r_next_state = IDLE;
    endcase
end

// ---- Registered outputs ------------------------------------------------------
// r_col_buf is read directly into a flop here, satisfying MEMORY_DFF.
always @(posedge clk) begin
    if (!rst_n) begin
        o_valid   <= 1'b0;
        o_ready   <= 1'b0;
        o_start   <= 1'b0;
        o_data    <= {DATA_WIDTH{1'b0}};
        o_ic_sel  <= {IC_SEL_W{1'b0}};
        o_ic_last <= 1'b0;
    end else begin
        o_valid   <= 1'b0;
        o_ready   <= 1'b0;
        o_start   <= 1'b0;
        o_data    <= {DATA_WIDTH{1'b0}};
        o_ic_sel  <= {IC_SEL_W{1'b0}};
        o_ic_last <= 1'b0;
        case (r_state)
            LEFT_PAD: begin
                o_valid   <= !r_stalled;
                o_start   <= (r_out_row == 0);
                o_ic_sel  <= r_ic_sel;
                o_ic_last <= ic_terminal;
                // Last beat before FILL: tell upstream to start sending.
                o_ready   <= col_done && last_pad_col;
            end

            FILL: begin
                o_ready <= !((r_fill_row == i_h_in - 1) && i_valid);
            end

            EMIT: begin
                o_valid   <= !r_stalled;
                o_start   <= (r_out_row == 0);
                o_ic_sel  <= r_ic_sel;
                o_ic_last <= ic_terminal;
                if (in_band) o_data <= r_col_buf[buf_idx];
                // Returning to FILL -- either for the next ic of this column,
                // or for the next real column.
                if (sweep_done && !(ic_terminal && last_real_col))
                    o_ready <= 1'b1;
            end

            RIGHT_PAD: begin
                o_valid   <= !r_stalled;
                o_start   <= (r_out_row == 0);
                o_ic_sel  <= r_ic_sel;
                o_ic_last <= ic_terminal;
            end
        endcase
    end
end

// ---- Position counters -------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        r_out_row  <= {ROW_W{1'b0}};
        r_out_col  <= {PADCOL_W{1'b0}};
        r_real_col <= {W_W{1'b0}};
    end else if (in_replay && i_ds_ready) begin
        if (r_out_row == padded_h - 1) begin
            r_out_row <= {ROW_W{1'b0}};
            // A row sweep finished. Only the LAST ic advances the column.
            if (ic_terminal) begin
                if (r_state == EMIT) begin
                    r_real_col <= r_real_col + 1'b1;
                    if (last_real_col) r_out_col <= {PADCOL_W{1'b0}};
                end else begin
                    // LEFT_PAD / RIGHT_PAD
                    r_out_col <= last_pad_col ? {PADCOL_W{1'b0}}
                                              : (r_out_col + 1'b1);
                end
            end
        end else begin
            r_out_row <= r_out_row + 1'b1;
        end
    end
end

// ---- Column fill -------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        r_fill_row <= {H_W{1'b0}};
    end else if (r_state == FILL && i_valid && (r_fill_row != 0 || i_start)) begin
        r_col_buf[r_fill_row] <= i_data;
        if (r_fill_row == i_h_in - 1) r_fill_row <= {H_W{1'b0}};
        else                          r_fill_row <= r_fill_row + 1'b1;
    end
end

endmodule
