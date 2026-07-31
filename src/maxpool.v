// =============================================================================
//  max_pool2d.v  —  Streaming 2×2 max pooling with stride 2
// =============================================================================
//
//  Performs 2×2 max pooling on a streaming feature map. Designed to integrate
//  seamlessly in a Conv → ReLU → Pool → ... pipeline with minimal latency.
//
//  Architecture
//  ------------
//  Two-stage comparison strategy:
//    1. Row pairs:    Compare even/odd rows (0↔1, 2↔3, ...) using r_row_hold
//    2. Column pairs: Compare even/odd columns using r_col_buf
//
//  Each pooled output is emitted IMMEDIATELY when its 2×2 window completes
//  (on odd row of odd column), not batched at end of column. This minimizes
//  per-cell latency for downstream pipeline stages.
//
//  Parameters
//  ----------
//  DATA_WIDTH : bit width of input/output data (typically 16 for int16 accum)
//  INPUT_HEIGHT          : input height (can be odd or even)
//  INPUT_WIDTH          : input width  (can be odd or even, tested up to INPUT_WIDTH=1000+)
//  POOL_HEIGHT     : output height = INPUT_HEIGHT/2 (integer division, truncates if INPUT_HEIGHT is odd)
//  POOL_WIDTH     : output width  = INPUT_WIDTH/2 (integer division, truncates if INPUT_WIDTH is odd)
//
//  Odd dimension handling
//  ----------------------
//  If INPUT_HEIGHT or INPUT_WIDTH is odd, the last row/column does not form a complete 2×2 window
//  and is TRUNCATED (not included in output). Examples:
//    5×5 input  →  2×2 output  (last row and column dropped)
//    6×8 input  →  3×4 output  (no truncation)
//
//  Memory footprint
//  ----------------
//  r_row_hold:  1 word           (even-row hold register)
//  r_col_buf:   POOL_HEIGHT words     (column buffer)
//  Total:     (POOL_HEIGHT + 1) words  —  constant regardless of INPUT_WIDTH
//
//  For INPUT_HEIGHT=1000 → POOL_HEIGHT=500 → 501 words = 1 KB (vs ~2 MB for naive buffering)
//
//  Latency
//  -------
//  First output:  Appears on odd row of column 1 (immediate emission)
//  Subsequent:    One output per odd row on odd columns (fully streaming)
//
//  Integration example
//  -------------------
//  conv_core → activation → max_pool2d → [next layer or output FIFO]
//
//  wire act_valid, pool_valid;
//  wire signed [15:0] act_data, pool_data;
//  ...
//  activation u_act (...);
//  max_pool2d u_pool (
//      .i_valid(act_valid), .i_data(act_data), ...
//      .o_valid(pool_valid), .o_data(pool_data), ...
//  );
//
// =============================================================================

module maxpool #(
    parameter DATA_WIDTH = 16,
    parameter INPUT_HEIGHT          = 8,
    parameter INPUT_WIDTH          = 100,
    parameter POOL_HEIGHT     = INPUT_HEIGHT / 2,
    parameter POOL_WIDTH      = INPUT_WIDTH  / 2,
    parameter ROW_BITS        = (POOL_HEIGHT > 1) ? $clog2(POOL_HEIGHT) : 1
)(
    input  wire                          clk,
    input  wire                          rst_n,
    
    // ── Input stream (from activation layer or conv core) ──
    input  wire                          i_valid,
    input  wire signed [DATA_WIDTH-1:0]  i_data,
    input  wire [$clog2(INPUT_HEIGHT)-1:0]          i_row,
    input  wire [$clog2(INPUT_WIDTH)-1:0]          i_col,
    
    // ── Pooled output stream ──
    output reg                           o_valid,
    output reg  signed [DATA_WIDTH-1:0]  o_data,
    output reg  [ROW_BITS-1:0]           o_row,
    output reg  [$clog2(POOL_WIDTH)-1:0] o_col
);

// =============================================================================
// Internal registers
// =============================================================================

// Row-hold register: stores even row for comparison with odd row
reg signed [DATA_WIDTH-1:0] r_row_hold;
reg                         r_row_hold_valid;

// Column buffer: stores row-pooled results from even columns
// Size = POOL_HEIGHT (independent of INPUT_WIDTH — key to memory efficiency)
reg signed [DATA_WIDTH-1:0] r_col_buf [0:POOL_HEIGHT-1];

// =============================================================================
// Combinational signals (decode input position)
// =============================================================================

wire r_even_row = (i_row[0] == 1'b0);
wire r_odd_row  = (i_row[0] == 1'b1);
wire r_even_col = (i_col[0] == 1'b0);
wire r_odd_col  = (i_col[0] == 1'b1);

// Output indices in pooled grid
wire [ROW_BITS-1:0] r_pool_row = i_row >> 1;
wire [$clog2(POOL_WIDTH)-1:0] r_pool_col = i_col >> 1;

// =============================================================================
// Max comparison function
// =============================================================================
function signed [DATA_WIDTH-1:0] max_of_two;
    input signed [DATA_WIDTH-1:0] a, b;
    begin
        max_of_two = (a > b) ? a : b;
    end
endfunction

reg signed [DATA_WIDTH-1:0]       r_col_buf1;
reg signed [DATA_WIDTH-1:0]       r_row_max;
reg                               r_output_valid;
reg [ROW_BITS-1:0]                r_pool_row_d;
reg [$clog2(POOL_WIDTH)-1:0]      r_pool_col_d;

wire signed [DATA_WIDTH-1:0] row_max = max_of_two(i_data, r_row_hold);

// r_output_valid must be recomputed every cycle (including i_valid low) so
// it correctly pulses low the cycle after a real sample even when input
// gaps follow -- it directly reflects this cycle's i_valid. The four data
// registers below it, though, are never consumed except on a cycle where
// r_output_valid is high, which only follows a cycle that had i_valid
// high -- so gating them with i_valid holds their last real capture
// unread during gaps, saving the r_col_buf read + comparator work on
// every idle cycle without changing what downstream ever observes.
always @(posedge clk) begin
    if (i_valid) begin
        r_col_buf1   <= r_col_buf[r_pool_row];
        r_row_max    <= row_max;
        r_pool_row_d <= r_pool_row;
        r_pool_col_d <= r_pool_col;
    end
    r_output_valid <= i_valid && r_odd_row && r_row_hold_valid && r_odd_col;
end

wire signed [DATA_WIDTH-1:0] pooled_max = max_of_two(r_row_max, r_col_buf1);

// =============================================================================
// Row-hold logic: separate always block for synthesis clarity
//
// Captures even rows (row 0, 2, 4, ...) for comparison with subsequent odd
// rows. The valid flag ensures we don't compare against stale data.
// =============================================================================
always @(posedge clk) begin
    if (!rst_n) begin
        r_row_hold       <= {DATA_WIDTH{1'b0}};
        r_row_hold_valid <= 1'b0;
    end else if (i_valid) begin
        if (r_even_row) begin
            // Even row: capture for next cycle's comparison
            r_row_hold       <= i_data;
            r_row_hold_valid <= 1'b1;
        end else begin
            // Odd row: clear valid flag (consumed this cycle)
            r_row_hold_valid <= 1'b0;
        end
    end
end

// =============================================================================
// Column buffer logic: separate always block
//
// On odd rows of even columns, store the row-pooled maximum for comparison
// when the corresponding odd column arrives. Buffer entry [i] holds the
// row-pooled result for output row i.
//
// Write timing: even_col && odd_row
// Read timing:  odd_col  && odd_row
// These are mutually exclusive → no read/write conflicts.
// =============================================================================
always @(posedge clk) begin
    if (!rst_n) begin
        integer i;
        for (i = 0; i < POOL_HEIGHT; i = i + 1)
            r_col_buf[i] <= {DATA_WIDTH{1'b0}};
    end else if (i_valid && r_odd_row && r_row_hold_valid && r_even_col) begin
        // Buffer the row-pooled result (even column, odd row)
        r_col_buf[r_pool_row] <= row_max;
    end
end

// =============================================================================
// Output logic: separate always block
//
// Emit pooled output immediately when a 2×2 window completes. This happens
// on odd rows of odd columns, when we have:
//   - Current sample (odd row, odd col)
//   - Row-paired sample (from r_row_hold)
//   - Column-buffered row-max (from r_col_buf)
//
// Timing: o_valid pulses high for exactly 1 cycle per pooled output.
// =============================================================================
always @(posedge clk) begin
    if (!rst_n) begin
        o_valid <= 1'b0;
        o_data  <= {DATA_WIDTH{1'b0}};
        o_row   <= {ROW_BITS{1'b0}};
        o_col   <= {$clog2(POOL_WIDTH){1'b0}};
    end else if (r_output_valid) begin
        // 2×2 window complete: emit pooled maximum
        o_valid <= 1'b1;
        o_data  <= pooled_max;
        o_row   <= r_pool_row_d;
        o_col   <= r_pool_col_d;
    end else begin
        o_valid <= 1'b0;
    end
end

// =============================================================================
// VCD dump (simulation only)
// =============================================================================
// initial begin
//     $dumpfile("dump.vcd");
//     $dumpvars(0, maxpool);
//     $dumpvars(0, r_row_hold);
//     $dumpvars(0, r_row_hold_valid);
//     for (integer i = 0; i < POOL_HEIGHT; i = i + 1)
//         $dumpvars(0, r_col_buf[i]);
// end

endmodule