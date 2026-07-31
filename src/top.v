// =============================================================================
//  top.v — TinyTapeout wrapper for the programmable convolution accelerator
// =============================================================================
//
//  Pin-compatible with ttsky-verilog-template's project.v:
//    ui_in[7:0]  uo_out[7:0]  uio_in/uio_out/uio_oe[7:0]  ena  clk  rst_n
//
//    uio_in[0]  mode       0 = PROGRAM (write config/weight/bias registers)
//                          1 = STREAM  (feed pixel columns)
//    uio_in[1]  in_valid   ui_in holds a byte this cycle (host-driven strobe)
//    uio_in[2]  out_ready  host can accept an output byte this cycle
//    uio_in[7:3]           unused (ignored)
//
//    uio_out[3] in_ready   this design can accept a byte on ui_in this cycle
//    uio_out[4] out_valid  uo_out holds a valid pooled-output byte this cycle
//    uio_out[5] busy       1 whenever mid-frame (STREAM / DRAIN)
//    uio_out[7:6], [2:0]   0
//
//  ─── 4-bit data over 8-bit pins ──────────────────────────────────────────
//  Samples and weights are DATA_WIDTH=4, so every pin byte carries TWO of
//  them: bits [3:0] first, bits [7:4] second. The datapath consumes one value
//  per cycle, so each accepted byte is unpacked into two consecutive internal
//  beats, with in_ready held low on the second so the host does not push
//  another byte. The SAME unpacker serves PROGRAM and STREAM, which is what
//  lets the register files keep a plain single-nibble write port.
//
//  ─── PROGRAM phase ────────────────────────────────────────────────────────
//  Registers are written once, in address order, starting at 0, so the address
//  never has to cross the pins: it is an internal counter (prog_addr, counting
//  NIBBLES) that resets every time the FSM (re-)enters PROGRAM. The image is:
//
//    nibble  0.. 19   config  H_IN, W_IN, IN_CHANNELS, PAD_W, PAD_H
//                             — 4 nibbles each, little-endian
//    nibble 20..      weights ic*K*K + kw*K + tap
//    nibble ..        bias    a single register
//
//  Boundaries come from the MAX_* envelope, not the programmed channel count,
//  so the host always writes TOTAL_REGS nibbles (ceil(TOTAL_REGS/2) bytes) and
//  simply writes don't-care values into unused channel slots. TOTAL_REGS may
//  be odd; the final byte's high nibble arrives with prog_addr == TOTAL_REGS,
//  where in_ready_prog gates the register write off and prog_addr holds.
//
//  ONE OUTPUT CHANNEL PER PASS. An output channel is an independent full
//  convolution, so the host reprograms this image and re-streams the image for
//  each channel it wants, stacking the resulting feature maps off-chip. That
//  is why there is no OUT_CHANNELS register and no channel index anywhere.
//
//  ─── STREAM phase ─────────────────────────────────────────────────────────
//  Byte order is column-major, input-channel in the middle, row innermost --
//  matching pad.v's column -> ic -> row replay nesting:
//
//    for col in 0..W_IN-1:
//        for ic in 0..IN_CHANNELS-1:
//            for row in 0,2,..H_IN-2:   send {pix[row+1][ic][col], pix[row][ic][col]}
//
//  H_IN MUST BE EVEN so a byte never straddles a channel or column boundary.
//
//  ─── Output ───────────────────────────────────────────────────────────────
//  There is no separate output phase: the pooled feature map is produced WHILE
//  input columns are still arriving, so the serializer runs continuously and
//  in parallel with STREAM. (in_valid and out_ready are independent uio_in
//  bits, so the pins are full duplex.) Each pooled value from the core is
//  latched and drained as ONE byte: the value in the low nibble, zero in the
//  high nibble. While a value is draining the core is back-pressured via
//  i_ds_ready -- the same handshake cascaded layers used to drive each other
//  with. DRAIN is only a tail state: it waits for the final value (detected at
//  pooled position POOL_H-1, POOL_W-1) to be accepted.
//
//  OUTPUT RATE. conv.v's emit side is deliberately free-running: no handshake
//  reaches it, because by the time a burst starts every cell it reads is
//  already computed, and stalling it would need backpressure inside the
//  circular accumulator where the read slot aliases the write slots every
//  KERNEL_SIZE columns. During an odd output column's burst maxpool therefore
//  produces a pooled value every 2 cycles and cannot be told to wait.
//
//  A FIFO deep enough for one whole burst absorbs that, and pad is released
//  only when the FIFO is empty -- so a burst can never start against a
//  partially-full FIFO and overflow is impossible by construction. The host
//  may take as long as it likes between out_ready pulses; it simply throttles
//  the input stream through the existing i_ds_ready path.
//
//  Starting a frame pulses a DATAPATH-ONLY reset (convCore's rst_n), leaving
//  the register files untouched via their separate reg_rst_n. That is what
//  lets frame N+1 run without reprogramming.
// =============================================================================

`timescale 1ns / 1ps

module top #(
    parameter DATA_WIDTH       = 4,     // nibble width; two per pin byte
    parameter KERNEL_SIZE      = 3,
    parameter MAX_H_IN         = 32,
    parameter MAX_W_IN         = 1024,
    parameter MAX_IN_CHANNELS  = 4,
    parameter MAX_PAD          = 1,
    parameter REG_ADDR_W       = 9      // must cover TOTAL_REGS
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       ena,

    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe
);

initial begin
    if (2*DATA_WIDTH != 8) begin
        $display("ERROR top: 2*DATA_WIDTH must be 8 -- ui_in/uo_out are fixed 8-bit pins");
        $finish;
    end
end

// =============================================================================
//  Derived local parameters — mirrors convCore's address map
// =============================================================================
localparam CFG_NIBBLES   = 20;
localparam TOTAL_REGS    = CFG_NIBBLES
                           + MAX_IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE
                           + 1;

localparam MAX_PADDED_HEIGHT = MAX_H_IN + 2*MAX_PAD;
localparam MAX_PADDED_WIDTH  = MAX_W_IN + 2*MAX_PAD;
localparam MAX_OUTPUT_HEIGHT = MAX_PADDED_HEIGHT - KERNEL_SIZE + 1;
localparam MAX_OUTPUT_WIDTH  = MAX_PADDED_WIDTH  - KERNEL_SIZE + 1;
localparam MAX_POOL_HEIGHT   = MAX_OUTPUT_HEIGHT / 2;
localparam MAX_POOL_WIDTH    = MAX_OUTPUT_WIDTH  / 2;

localparam H_W           = $clog2(MAX_H_IN + 1);
localparam W_W           = $clog2(MAX_W_IN + 1);
localparam IC_W          = $clog2(MAX_IN_CHANNELS + 1);
localparam PAD_W_W       = $clog2(MAX_PAD + 1);
localparam POOL_ROW_BITS = (MAX_POOL_HEIGHT > 1) ? $clog2(MAX_POOL_HEIGHT) : 1;
localparam POOL_COL_BITS = $clog2(MAX_POOL_WIDTH);

// =============================================================================
//  uio_* decode / encode
// =============================================================================
wire mode      = uio_in[0];   // 0 = PROGRAM, 1 = STREAM
wire in_valid  = uio_in[1];
wire out_ready = uio_in[2];

assign uio_oe = ena ? 8'b0011_1000 : 8'b0000_0000;   // bits [5:3] are outputs

// =============================================================================
//  FSM states
// =============================================================================
localparam [1:0]
    PROG   = 2'd0,
    STREAM = 2'd1,
    DRAIN  = 2'd2;

reg [1:0]              state;
reg [REG_ADDR_W-1:0]   prog_addr;
reg [H_W-1:0]          row_cnt;
reg [IC_W-1:0]         ic_cnt;
reg [W_W-1:0]          col_cnt;

// Datapath-only reset, pulsed when a frame starts. Register files sit on
// reg_rst_n and are untouched, so a second frame needs no reprogramming.
reg [1:0] r_core_rst;
wire      core_rst_n = rst_n && (r_core_rst == 2'd0);

// ---- Nibble unpacker --------------------------------------------------------
reg                  r_half;      // 1 = the held high nibble is due this cycle
reg [DATA_WIDTH-1:0] r_byte_hi;

// =============================================================================
//  Core interface
// =============================================================================
wire                                          core_o_ready;
wire                                          core_o_valid;
wire signed [DATA_WIDTH-1:0]                  core_o_data;
wire [POOL_ROW_BITS-1:0]                      core_o_row;
wire [POOL_COL_BITS-1:0]                      core_o_col;

wire [H_W-1:0]     cfg_h_in;
wire [W_W-1:0]     cfg_w_in;
wire [IC_W-1:0]    cfg_in_channels;
wire [PAD_W_W-1:0] cfg_pad_w;
wire [PAD_W_W-1:0] cfg_pad_h;

// ---- Pooled-value FIFO ------------------------------------------------------
// conv.v's emit side is free-running by design: once a burst starts it walks a
// whole output column and no handshake can stop it (stalling it would need
// backpressure inside the circular accumulator, where the read slot aliases
// the write slots every KERNEL_SIZE columns). maxpool fires on odd rows of odd
// columns, so a burst delivers up to MAX_POOL_HEIGHT values, one every 2
// cycles. This FIFO absorbs a whole burst so the host can drain at its own
// pace instead of having to keep up cycle by cycle.
//
// Depth is rounded up to a power of two so the pointers wrap for free, and is
// therefore always at least one full burst. pad is released only when the FIFO
// is EMPTY, which is what makes overflow impossible by construction rather
// than by timing margin: a burst only starts when a real input column lands,
// and no column can land while pad is held.
// The floor of 1 keeps the pointer slice legal on a tiny envelope, matching
// how POOL_ROW_BITS guards itself.
localparam OUT_FIFO_AW    = ($clog2(MAX_POOL_HEIGHT) > 1) ? $clog2(MAX_POOL_HEIGHT) : 1;
localparam OUT_FIFO_DEPTH = 1 << OUT_FIFO_AW;

// Each entry carries its end-of-frame marker alongside the value, so DRAIN
// still sees "last" attached to the right value however long it sat queued.
reg [DATA_WIDTH:0]  fifo_mem [0:OUT_FIFO_DEPTH-1];
reg [OUT_FIFO_AW:0] fifo_wptr, fifo_rptr;   // one extra bit separates full
                                            // from empty

wire fifo_empty = (fifo_wptr == fifo_rptr);
wire fifo_full  = (fifo_wptr[OUT_FIFO_AW] != fifo_rptr[OUT_FIFO_AW]) &&
                  (fifo_wptr[OUT_FIFO_AW-1:0] == fifo_rptr[OUT_FIFO_AW-1:0]);

// ---- Output byte serializer -------------------------------------------------
// One pooled value per byte. The output register is the FIFO's final stage: it
// reloads whenever it is free, or is being freed by the host this cycle.
reg                    out_busy;
reg [DATA_WIDTH-1:0]   out_shift;
reg                    out_last;

wire out_valid_w = out_busy;
wire fifo_rd     = (!out_busy || out_ready) && !fifo_empty;

// Pooled-map extent, from the same registers the datapath runs on.
wire [POOL_ROW_BITS-1:0] pool_h_m1 =
    (((cfg_h_in + {cfg_pad_h, 1'b0}) - (KERNEL_SIZE - 1)) >> 1) - 1;
wire [POOL_COL_BITS-1:0] pool_w_m1 =
    (((cfg_w_in + {cfg_pad_w, 1'b0}) - (KERNEL_SIZE - 1)) >> 1) - 1;

wire value_is_last = (core_o_row == pool_h_m1) && (core_o_col == pool_w_m1);
wire frame_done    = out_busy && out_last && out_ready;

// ---- Handshakes -------------------------------------------------------------
wire in_ready_prog   = (prog_addr < TOTAL_REGS);
// row_cnt != 0 means we are mid-column: pad accepts the rest of a column
// without renewed backpressure, mirroring its FILL behaviour.
wire in_ready_stream = (r_core_rst == 2'd0) && ((row_cnt != 0) || core_o_ready);

wire in_ready = !r_half &&
                ((state == PROG)   ? in_ready_prog   :
                 (state == STREAM) ? in_ready_stream :
                                     1'b0);

wire byte_accept = in_valid && in_ready;
// One internal beat per nibble: the accepted byte's low half this cycle, its
// high half the next (during which in_ready is low).
wire                  nib_valid = byte_accept || r_half;
wire [DATA_WIDTH-1:0] nib_data  = r_half ? r_byte_hi : ui_in[DATA_WIDTH-1:0];

wire core_valid  = (state == STREAM) && nib_valid;
wire core_start  = core_valid && (row_cnt == {H_W{1'b0}});
wire reg_wr_en_w = (state == PROG) && nib_valid && in_ready_prog;
wire busy        = (state != PROG);

convCore #(
    .DATA_WIDTH       (DATA_WIDTH),
    .KERNEL_SIZE      (KERNEL_SIZE),
    .MAX_H_IN         (MAX_H_IN),
    .MAX_W_IN         (MAX_W_IN),
    .MAX_IN_CHANNELS  (MAX_IN_CHANNELS),
    .MAX_PAD          (MAX_PAD),
    .REG_ADDR_W       (REG_ADDR_W)
) u_convCore (
    .clk              (clk),
    .rst_n            (core_rst_n),
    .reg_rst_n        (rst_n),

    // pixel stream
    .i_valid          (core_valid),
    .start            (core_start),
    .i_data           (nib_data),
    .o_ready          (core_o_ready),

    // register programming — address generated internally (prog_addr)
    .reg_wr_en        (reg_wr_en_w),
    .reg_wr_addr      (prog_addr),
    .reg_wr_data      (nib_data),

    // pooled output + backpressure from the output FIFO below. Holding pad
    // until the FIFO is fully drained is what guarantees a whole emit burst
    // always fits (see the FIFO comment above).
    .i_ds_ready       (fifo_empty),
    .o_valid          (core_o_valid),
    .o_data           (core_o_data),
    .o_row            (core_o_row),
    .o_col            (core_o_col),

    .cfg_h_in         (cfg_h_in),
    .cfg_w_in         (cfg_w_in),
    .cfg_in_channels  (cfg_in_channels),
    .cfg_pad_w        (cfg_pad_w),
    .cfg_pad_h        (cfg_pad_h)
);

// =============================================================================
//  Nibble unpacker
// =============================================================================
always @(posedge clk) begin
    if (!rst_n) begin
        r_half    <= 1'b0;
        r_byte_hi <= {DATA_WIDTH{1'b0}};
    end else if (byte_accept) begin
        r_half    <= 1'b1;
        r_byte_hi <= ui_in[2*DATA_WIDTH-1 -: DATA_WIDTH];
    end else if (r_half) begin
        r_half    <= 1'b0;
    end
end

// =============================================================================
//  Main FSM
// =============================================================================
always @(posedge clk) begin
    if (!rst_n) begin
        state      <= PROG;
        prog_addr  <= {REG_ADDR_W{1'b0}};
        row_cnt    <= {H_W{1'b0}};
        ic_cnt     <= {IC_W{1'b0}};
        col_cnt    <= {W_W{1'b0}};
        r_core_rst <= 2'd0;
    end else begin
        if (r_core_rst != 2'd0) r_core_rst <= r_core_rst - 1'b1;

        case (state)
            // ------------------------------------------------------------------
            //  Write config/weight/bias registers in address order from 0.
            //  Raising mode ends programming and starts a frame.
            PROG: begin
                if (mode) begin
                    state      <= STREAM;
                    row_cnt    <= {H_W{1'b0}};
                    ic_cnt     <= {IC_W{1'b0}};
                    col_cnt    <= {W_W{1'b0}};
                    r_core_rst <= 2'd3;      // restart the datapath, keep the regs
                end else if (nib_valid && in_ready_prog) begin
                    prog_addr <= prog_addr + 1'b1;
                end
            end

            // ------------------------------------------------------------------
            //  Stream W_IN columns, each IN_CHANNELS passes of H_IN rows.
            //  Dropping mode aborts the frame and returns to PROGRAM.
            STREAM: begin
                if (!mode) begin
                    state     <= PROG;
                    prog_addr <= {REG_ADDR_W{1'b0}};
                end else if (core_valid) begin
                    if (row_cnt == cfg_h_in - 1) begin
                        row_cnt <= {H_W{1'b0}};
                        if (ic_cnt == cfg_in_channels - 1) begin
                            ic_cnt <= {IC_W{1'b0}};
                            if (col_cnt == cfg_w_in - 1) begin
                                col_cnt <= {W_W{1'b0}};
                                state   <= DRAIN;
                            end else begin
                                col_cnt <= col_cnt + 1'b1;
                            end
                        end else begin
                            ic_cnt <= ic_cnt + 1'b1;
                        end
                    end else begin
                        row_cnt <= row_cnt + 1'b1;
                    end
                end
            end

            // ------------------------------------------------------------------
            //  All input sent. The serializer is still running; wait for the
            //  final pooled bundle to be accepted, then start the next frame.
            DRAIN: begin
                if (frame_done) begin
                    if (mode) begin
                        state      <= STREAM;
                        row_cnt    <= {H_W{1'b0}};
                        ic_cnt     <= {IC_W{1'b0}};
                        col_cnt    <= {W_W{1'b0}};
                        r_core_rst <= 2'd3;
                    end else begin
                        state     <= PROG;
                        prog_addr <= {REG_ADDR_W{1'b0}};
                    end
                end
            end

            default: state <= PROG;
        endcase
    end
end

// =============================================================================
//  Output FIFO write port — captures the free-running emit burst
// =============================================================================
// The !fifo_full guard is defensive only: the fifo_empty gate on pad makes
// overflow unreachable. If it ever did trigger, dropping a value beats
// corrupting the pointers and desynchronising the whole frame.
// Reset with the DATAPATH (core_rst_n), not the chip: starting a frame must
// discard anything left queued from an aborted one, or stale values would head
// the next frame's stream. A mode drop to PROGRAM leaves the queue dirty, but
// raising mode again pulses r_core_rst before any new value can be produced.
always @(posedge clk) begin
    if (!core_rst_n) begin
        fifo_wptr <= {(OUT_FIFO_AW+1){1'b0}};
    end else if (core_o_valid && !fifo_full) begin
        fifo_mem[fifo_wptr[OUT_FIFO_AW-1:0]] <= {value_is_last, core_o_data};
        fifo_wptr <= fifo_wptr + 1'b1;
    end
end

// =============================================================================
//  Output byte serializer — runs continuously, concurrent with STREAM
// =============================================================================
always @(posedge clk) begin
    if (!core_rst_n) begin
        fifo_rptr <= {(OUT_FIFO_AW+1){1'b0}};
        out_busy  <= 1'b0;
        out_shift <= {DATA_WIDTH{1'b0}};
        out_last  <= 1'b0;
    end else if (fifo_rd) begin
        // Reload in the SAME cycle the host takes the current byte, so a
        // fully-ready host still drains one value per cycle.
        {out_last, out_shift} <= fifo_mem[fifo_rptr[OUT_FIFO_AW-1:0]];
        fifo_rptr <= fifo_rptr + 1'b1;
        out_busy  <= 1'b1;
    end else if (out_busy && out_ready) begin
        out_busy <= 1'b0;
    end
end

// =============================================================================
//  Output pins
// =============================================================================
// One pooled value per byte: the value in [3:0], zero in [7:4].
wire [2*DATA_WIDTH-1:0] out_byte = {{DATA_WIDTH{1'b0}}, out_shift};

assign uo_out  = (ena && out_valid_w) ? out_byte : 8'b0;
assign uio_out = ena ? {2'b00, busy, out_valid_w, in_ready, 3'b000} : 8'b0;

endmodule
