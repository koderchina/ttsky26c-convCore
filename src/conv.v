// =============================================================================
//  conv.v  —  streaming 2-D convolution core (multi-input-channel accumulating)
// =============================================================================
//
//  Three PEs (one per kernel column) feed a small CIRCULAR accumulator:
//  KERNEL_SIZE column-slots per output row, never a whole feature map, because
//  a convolution window is only open for KERNEL_SIZE input columns.
//
//  ONE replica axis collapses onto this pipeline. pad.v presents each column
//  position as an ic x row sweep and tags every beat with:
//
//    ic_sel_in    -> which input channel's weight slice to multiply by
//    ic_last_in   -> "this beat is from the final input channel"
//
//  The accumulator address is (row, slot) with NO channel dimension: every ic
//  pass over new data simply += into the same cell. That is precisely
//  "accumulate the intermediate results of all input channels", using the
//  buffer that already existed.
//
//  There is no output-channel axis on-chip at all. An output channel is an
//  independent full convolution, so the host programs one channel's weights,
//  streams the image, collects the resulting feature map, and repeats. That is
//  what keeps this accumulator at KERNEL_SIZE banks of MAX_OUTPUT_HEIGHT words
//  each (one bank per slot, addressed by row alone) -- the previous
//  (oc,row,slot) flat form was what made the decode logic dominate the die and
//  saturate routing.
//
//  ic_last_in only gates the two places that mean "a real input column has
//  fully landed": the input-column counter and the emit trigger. Without it
//  those would fire once per input channel instead of once per column.
//
//  Geometry is RUNTIME (i_padded_w / i_padded_h); the MAX_* parameters bound
//  array and bus widths. Accumulator address arithmetic deliberately uses the
//  COMPILE-TIME KERNEL_SIZE stride -- the array is sized for the worst case
//  regardless, so a runtime stride would synthesize a multiplier for nothing.
//
// =============================================================================

`timescale 1ns/1ps

module conv #(
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

    // Worst-case magnitude in one cell is K*K*MAX_IN_CHANNELS products of two
    // DATA_WIDTH signed values, i.e. K*K*MAX_IN_CHANNELS * 2^(2*DW-2). The
    // +1 for the sign is folded into the 2*DW-1 term. Derived rather than
    // fixed so raising MAX_IN_CHANNELS cannot silently start wrapping.
    localparam ACC_WIDTH = 2*DATA_WIDTH - 1
                           + $clog2(KERNEL_SIZE*KERNEL_SIZE*MAX_IN_CHANNELS),

    localparam IC_SEL_W  = (MAX_IN_CHANNELS <= 1) ? 1 : $clog2(MAX_IN_CHANNELS),

    localparam PADH_W    = $clog2(MAX_PADDED_HEIGHT + 1),
    localparam PADW_W    = $clog2(MAX_PADDED_WIDTH  + 1),
    localparam OUTH_W    = $clog2(MAX_OUTPUT_HEIGHT),
    localparam OUTW_W    = $clog2(MAX_OUTPUT_WIDTH)
)(
    input  wire clk,
    input  wire rst_n,

    // ---- Runtime geometry ---------------------------------------------------
    input  wire [PADW_W-1:0] i_padded_w,
    input  wire [PADH_W-1:0] i_padded_h,

    // ---- Sample stream from pad --------------------------------------------
    input  wire                                     start,
    input  wire                                     i_valid,
    input  wire signed [DATA_WIDTH-1:0]             i_data,
    input  wire [MAX_IN_CHANNELS*KERNEL_SIZE*KERNEL_SIZE*DATA_WIDTH-1:0] weights_in,
    // Which input channel's weight slice this beat multiplies by.
    input  wire [IC_SEL_W-1:0]                      ic_sel_in,
    // This beat belongs to the final input channel of the current column.
    input  wire                                     ic_last_in,

    output reg                                      o_valid,
    output reg [OUTW_W-1:0]                         o_col,
    output reg [OUTH_W-1:0]                         o_row,
    output reg signed [ACC_WIDTH-1:0]               o_data,
    output wire                                     done_flag
);

// ---- Runtime output geometry -------------------------------------------------
wire [PADW_W-1:0] output_w = i_padded_w - (KERNEL_SIZE - 1);
wire [PADH_W-1:0] output_h = i_padded_h - (KERNEL_SIZE - 1);

// ---------------------------------------------------------------------------
// Declarations — all regs/wires declared before any always block
// ---------------------------------------------------------------------------
localparam START = 0;
localparam RUN   = 2;
localparam DONE  = 3;

reg [1:0] r_state, r_next_state;

reg [PADW_W-1:0]     r_input_col_counter;
reg [OUTH_W-1:0]     r_out_row;

// Circular accumulator — three discrete banks, one per slot; see "Circular
// accumulator" below for why.
reg signed [ACC_WIDTH-1:0] r_accum0 [0:MAX_OUTPUT_HEIGHT-1];
reg signed [ACC_WIDTH-1:0] r_accum1 [0:MAX_OUTPUT_HEIGHT-1];
reg signed [ACC_WIDTH-1:0] r_accum2 [0:MAX_OUTPUT_HEIGHT-1];
integer bank_row;

// Weight slice for this cycle, feeding all 3 PEs with the same K*K*DATA_WIDTH
// layout each PE already expected. Selected by the input-channel index, so
// multi-channel weights flow straight from the regfile with no repacking.
wire [KERNEL_SIZE*KERNEL_SIZE*DATA_WIDTH-1:0] weights_cur =
    weights_in[ic_sel_in*KERNEL_SIZE*KERNEL_SIZE*DATA_WIDTH +: KERNEL_SIZE*KERNEL_SIZE*DATA_WIDTH];

// pe.v's o_data/o_valid/done are REGISTERED -- they reflect the ic_last_in
// from ONE CYCLE AGO, not the current one. This delayed copy tracks that lag
// so the end-of-column gating matches the beat the just-arrived data actually
// belongs to. Invisible at IN_CHANNELS<=1, which is why single-channel tests
// never caught the equivalent bug.
reg r_ic_last_prev;
always @(posedge clk) begin
    if (!rst_n) r_ic_last_prev <= 1'b0;
    else        r_ic_last_prev <= ic_last_in;
end

reg                  emitting;
reg [OUTH_W-1:0]     emit_row;
reg [OUTW_W-1:0]     emit_col;

reg r_prev_done;

wire pe_o_valid_0, pe_o_valid_1, pe_o_valid_2;
wire signed [ACC_WIDTH-1:0] o_data_0, o_data_1, o_data_2;
wire done_0, done_1, done_2;

// "A real input column has fully landed" -- once per column, not once per
// input channel. This is the ONLY structural change the ic axis requires.
wire col_complete = done_0 && r_ic_last_prev;

// ---------------------------------------------------------------------------
// FSM — START → RUN → DONE
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) r_state <= START;
    else        r_state <= r_next_state;
end

always @(*) begin
    r_next_state = r_state;
    case (r_state)
        START:   r_next_state = RUN;
        RUN:     if (r_input_col_counter == i_padded_w && !emitting) r_next_state = DONE;
        DONE:    ; // park until reset
        default: r_next_state = START;
    endcase
end

// ---------------------------------------------------------------------------
// Input-column counter — advances once per real column, after the last ic pass
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n)
        r_input_col_counter <= {PADW_W{1'b0}};
    else if (r_state == RUN && col_complete)
        r_input_col_counter <= r_input_col_counter + 1'b1;
end

// ---------------------------------------------------------------------------
// Circular slot wires
// ---------------------------------------------------------------------------
wire [1:0] slot0 = r_input_col_counter % KERNEL_SIZE;
wire [1:0] slot1 = (r_input_col_counter >= 1) ? (r_input_col_counter - 1) % KERNEL_SIZE : 0;
wire [1:0] slot2 = (r_input_col_counter >= 2) ? (r_input_col_counter - 2) % KERNEL_SIZE : 0;

// ---------------------------------------------------------------------------
// PE valid gating
// ---------------------------------------------------------------------------
wire pe_i_valid_0 = i_valid && (r_state == RUN);
wire pe_i_valid_1 = i_valid && (r_state == RUN) && (r_input_col_counter >= 1);
wire pe_i_valid_2 = i_valid && (r_state == RUN) && (r_input_col_counter >= 2);

wire we0 = pe_o_valid_0 && (r_input_col_counter       < output_w);
wire we1 = pe_o_valid_1 && (r_input_col_counter >= 1) && (r_input_col_counter - 1 < output_w);
wire we2 = pe_o_valid_2 && (r_input_col_counter >= 2) && (r_input_col_counter - 2 < output_w);

// ---------------------------------------------------------------------------
// Per-PE start strobes
// ---------------------------------------------------------------------------
reg start0, start1, start2;
always @(*) begin
    start0 = 0; start1 = 0; start2 = 0;
    if (r_state == RUN) begin
        start0 = start;
        if (r_input_col_counter >= 1) start1 = start;
        if (r_input_col_counter >= 2) start2 = start;
    end
end

// ---------------------------------------------------------------------------
// Output-row counter
//
// `start` fires at row 0 of EVERY presentation -- including every new ic pass
// -- so this resets per channel pass with no ic awareness needed.
//
// The increment saturates at MAX_OUTPUT_HEIGHT-1 so this stays a valid r_accum
// address at all times. pe.v emits exactly output_h valid pulses per
// presentation, so the live addresses are 0..output_h-1 and the counter is
// never asked to reach output_h while a write is enabled. It does reach it
// otherwise: pad.v goes EMIT -> FILL -> EMIT between input channels, and the
// registered pe.o_valid lands one cycle after a presentation's last beat, with
// no coinciding `start` to clear the counter. At the maximum geometry that
// terminal value is MAX_OUTPUT_HEIGHT itself, one past the end of the banks.
//
// Nothing reads it there -- every use sits inside an `if (bankN_weM)`, all of
// which are low across the FILL gap -- but leaving it unclamped made the
// counter one bit wider than the banks it addresses, and Yosys silently drops
// that bit ("Removed top 1 address bits (of 6)"). RTL then reads X where the
// netlist reads row 0. Harmless while the value is dead, and a trap for
// whoever next uses that read outside the write-enable guard.
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        r_out_row <= {OUTH_W{1'b0}};
    end else if (r_state == RUN) begin
        if (start)
            r_out_row <= {OUTH_W{1'b0}};
        else if (pe_o_valid_0 && r_out_row != MAX_OUTPUT_HEIGHT - 1)
            r_out_row <= r_out_row + 1'b1;
    end
end

// ---------------------------------------------------------------------------
// Circular accumulator — one discrete register bank per slot instead of one
// flat MAX_OUTPUT_HEIGHT*KERNEL_SIZE array addressed by row*KERNEL_SIZE+slot.
//
// slot0/1/2 are three consecutive values mod KERNEL_SIZE, so whenever all
// three are active they are always a permutation of {0,1,2} -- each bank
// receives from at most ONE of we0/we1/we2 on any given cycle. Which PE
// feeds a given bank is therefore a fixed rotation resolved by comparing
// slotN against that bank's own fixed number (a small mux ahead of a
// MAX_OUTPUT_HEIGHT-word array), not a decode across the full row*KERNEL_SIZE
// address space every write previously had to clear.
//
// The clear is NOT mutually exclusive with the accumulates, though: it
// addresses emit_row while they address r_out_row, so a bank can legitimately
// take a clear and an accumulate in the same cycle, at two different rows.
// Both writes have to happen -- see the write block below.
// ---------------------------------------------------------------------------
wire [1:0] emit_slot = emit_col % KERNEL_SIZE;

wire bank0_clear = emitting && (emit_col + KERNEL_SIZE) < output_w && (emit_slot == 2'd0);
wire bank0_we0   = we0 && (slot0 == 2'd0);
wire bank0_we1   = we1 && (slot1 == 2'd0);
wire bank0_we2   = we2 && (slot2 == 2'd0);

wire bank1_clear = emitting && (emit_col + KERNEL_SIZE) < output_w && (emit_slot == 2'd1);
wire bank1_we0   = we0 && (slot0 == 2'd1);
wire bank1_we1   = we1 && (slot1 == 2'd1);
wire bank1_we2   = we2 && (slot2 == 2'd1);

wire bank2_clear = emitting && (emit_col + KERNEL_SIZE) < output_w && (emit_slot == 2'd2);
wire bank2_we0   = we0 && (slot0 == 2'd2);
wire bank2_we1   = we1 && (slot1 == 2'd2);
wire bank2_we2   = we2 && (slot2 == 2'd2);

always @(posedge clk) begin
    if (!rst_n) begin
        for (bank_row = 0; bank_row < MAX_OUTPUT_HEIGHT; bank_row = bank_row + 1) begin
            r_accum0[bank_row] <= 0;
            r_accum1[bank_row] <= 0;
            r_accum2[bank_row] <= 0;
        end
    end else if (r_state == RUN) begin
        // These MUST stay separate `if`s, not a case. The clear writes
        // emit_row while the accumulates write r_out_row -- different
        // addresses, so in any cycle where both fire they are two independent
        // writes to one bank, and a case (which selects exactly one branch)
        // silently drops the clear. That leaves a stale sum in the slot and
        // the next column reusing it accumulates on top of garbage. Ordering
        // matters only for a true same-address collision (r_out_row ==
        // emit_row): clear first, accumulate last, so the accumulate wins --
        // matching the flat array this replaced.
        if (bank0_clear) r_accum0[emit_row]  <= 0;
        if (bank0_we0)   r_accum0[r_out_row] <= r_accum0[r_out_row] + o_data_0;
        if (bank0_we1)   r_accum0[r_out_row] <= r_accum0[r_out_row] + o_data_1;
        if (bank0_we2)   r_accum0[r_out_row] <= r_accum0[r_out_row] + o_data_2;

        if (bank1_clear) r_accum1[emit_row]  <= 0;
        if (bank1_we0)   r_accum1[r_out_row] <= r_accum1[r_out_row] + o_data_0;
        if (bank1_we1)   r_accum1[r_out_row] <= r_accum1[r_out_row] + o_data_1;
        if (bank1_we2)   r_accum1[r_out_row] <= r_accum1[r_out_row] + o_data_2;

        if (bank2_clear) r_accum2[emit_row]  <= 0;
        if (bank2_we0)   r_accum2[r_out_row] <= r_accum2[r_out_row] + o_data_0;
        if (bank2_we1)   r_accum2[r_out_row] <= r_accum2[r_out_row] + o_data_1;
        if (bank2_we2)   r_accum2[r_out_row] <= r_accum2[r_out_row] + o_data_2;
    end
end

// ---------------------------------------------------------------------------
// Streaming output — triggered once per REAL column (after the last ic pass)
//
// The emit side is independent of the input pacing: by the time a burst
// starts, every cell it will read is fully computed, so emission runs at full
// speed regardless of upstream FILL gaps.
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        emitting  <= 1'b0;
        emit_row  <= {OUTH_W{1'b0}};
        emit_col  <= {OUTW_W{1'b0}};
    end else begin
        if (col_complete && r_input_col_counter >= KERNEL_SIZE - 1) begin
            emitting  <= 1'b1;
            emit_row  <= {OUTH_W{1'b0}};
            emit_col  <= r_input_col_counter - (KERNEL_SIZE - 1);
        end else if (emitting) begin
            if (emit_row == output_h - 1) emitting <= 1'b0;
            else                          emit_row <= emit_row + 1'b1;
        end
    end
end

// o_valid must track `emitting` unconditionally (it has to fall the cycle
// emitting drops). The data fields are only ever read downstream on a cycle
// where o_valid is high, so gating them with `emitting` skips the bank read
// on idle cycles without changing what downstream observes.
always @(posedge clk) begin
    o_valid <= emitting;
    if (emitting) begin
        o_row  <= emit_row;
        o_col  <= emit_col;
        case (emit_slot)
            2'd0:    o_data <= r_accum0[emit_row];
            2'd1:    o_data <= r_accum1[emit_row];
            default: o_data <= r_accum2[emit_row];
        endcase
    end
end

// ---------------------------------------------------------------------------
// Done flag
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) r_prev_done <= 1'b0;
    else        r_prev_done <= (r_state == DONE);
end
assign done_flag = (r_state == DONE) && !r_prev_done;

// ---------------------------------------------------------------------------
// PE instantiations — each PE gets its KERNEL_SIZE weights from weights_cur.
// weights_cur layout: [kw=0 taps | kw=1 taps | kw=2 taps]
//   each group: tap0 at LSB (newest sample) .. tap K-1 at MSB (oldest)
//
// new_sample is tied high: with the oc replica axis gone, every presented beat
// is a fresh sample rather than a reweigh of held data.
// ---------------------------------------------------------------------------
pe #(.DATA_WIDTH(DATA_WIDTH), .MAX_ROWS(MAX_PADDED_HEIGHT), .KERNEL_SIZE(KERNEL_SIZE), .ACC_WIDTH(ACC_WIDTH)) u_pe_0 (
    .clk        (clk),
    .rst_n      (rst_n),
    .i_valid    (pe_i_valid_0),
    .i_rows     (i_padded_h),
    .new_sample (1'b1),
    .start      (start0),
    .i_data     (i_data),
    .weights_in (weights_cur[0*KERNEL_SIZE*DATA_WIDTH +: KERNEL_SIZE*DATA_WIDTH]),
    .o_data     (o_data_0),
    .done       (done_0),
    .o_valid    (pe_o_valid_0)
);

pe #(.DATA_WIDTH(DATA_WIDTH), .MAX_ROWS(MAX_PADDED_HEIGHT), .KERNEL_SIZE(KERNEL_SIZE), .ACC_WIDTH(ACC_WIDTH)) u_pe_1 (
    .clk        (clk),
    .rst_n      (rst_n),
    .i_valid    (pe_i_valid_1),
    .i_rows     (i_padded_h),
    .new_sample (1'b1),
    .start      (start1),
    .i_data     (i_data),
    .weights_in (weights_cur[1*KERNEL_SIZE*DATA_WIDTH +: KERNEL_SIZE*DATA_WIDTH]),
    .o_data     (o_data_1),
    .done       (done_1),
    .o_valid    (pe_o_valid_1)
);

pe #(.DATA_WIDTH(DATA_WIDTH), .MAX_ROWS(MAX_PADDED_HEIGHT), .KERNEL_SIZE(KERNEL_SIZE), .ACC_WIDTH(ACC_WIDTH)) u_pe_2 (
    .clk        (clk),
    .rst_n      (rst_n),
    .i_valid    (pe_i_valid_2),
    .i_rows     (i_padded_h),
    .new_sample (1'b1),
    .start      (start2),
    .i_data     (i_data),
    .weights_in (weights_cur[2*KERNEL_SIZE*DATA_WIDTH +: KERNEL_SIZE*DATA_WIDTH]),
    .o_data     (o_data_2),
    .done       (done_2),
    .o_valid    (pe_o_valid_2)
);

endmodule
