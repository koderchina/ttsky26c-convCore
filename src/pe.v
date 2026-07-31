`timescale 1ns/1ps

// =============================================================================
//  pe.v  —  1-D column convolution processing element
// =============================================================================
//
//  Weights are supplied combinatorially via weights_in; no internal weight
//  shift register or prog mode.  weights_in[k*DATA_WIDTH +: DATA_WIDTH] is
//  tap k, where tap 0 multiplies the current (newest) sample.
//
// =============================================================================

module pe #(
    parameter DATA_WIDTH  = 4,
    parameter MAX_ROWS    = 34,    // worst-case rows in a padded input column
    parameter KERNEL_SIZE = 3,
    parameter ACC_WIDTH   = 13,
    localparam ROW_CNT_W  = $clog2(MAX_ROWS + 1)
)(
    input  wire                                     clk,
    input  wire                                     rst_n,
    input  wire                                     i_valid,
    input  wire                                     start,
    // Rows in the padded column currently being streamed. Runtime, not a
    // parameter: the column height follows H_IN/PAD_H, which are programmed
    // at run time now. Only `done` below depends on it.
    input  wire [ROW_CNT_W-1:0]                     i_rows,
    // High on the LAST replica cycle of the current pixel's replay group
    // (the cycle immediately before i_data changes to the next row's
    // value) -- NOT the first. The hist shift below takes effect via NBA,
    // visible starting next cycle, so triggering it on the last replica
    // is what makes every replica of a row observe identical history.
    // Low on the earlier replay cycles, where the same pixel is
    // re-presented against a different output channel's weights_in.
    input  wire                                     new_sample,
    input  wire signed [DATA_WIDTH-1:0]             i_data,
    input  wire        [KERNEL_SIZE*DATA_WIDTH-1:0] weights_in,
    output reg  signed [ACC_WIDTH-1:0]              o_data,
    output reg                                      done,
    output reg                                      o_valid
);

// Signed aliases for weight taps
wire signed [DATA_WIDTH-1:0] w [0:KERNEL_SIZE-1];
genvar gk;
generate
    for (gk = 0; gk < KERNEL_SIZE; gk = gk + 1) begin : g_w
        assign w[gk] = $signed(weights_in[gk*DATA_WIDTH +: DATA_WIDTH]);
    end
endgenerate

// Input history: hist[0]=1-ago, hist[1]=2-ago
reg signed [DATA_WIDTH-1:0]    hist         [0:KERNEL_SIZE-2];
reg        [KERNEL_SIZE-2:0]   r_hist_valid;
reg        [ROW_CNT_W-1:0]     r_row_count;

always @(posedge clk) begin
    if (!rst_n) begin
        hist[0]      <= {DATA_WIDTH{1'b0}};
        hist[1]      <= {DATA_WIDTH{1'b0}};
        r_hist_valid <= {(KERNEL_SIZE-1){1'b0}};
        r_row_count  <= {ROW_CNT_W{1'b0}};
    end else if (i_valid && new_sample) begin
        if (start) begin
            hist[0]      <= i_data;
            hist[1]      <= {DATA_WIDTH{1'b0}};
            r_hist_valid <= {{(KERNEL_SIZE-2){1'b0}}, 1'b1};
            r_row_count  <= {ROW_CNT_W{1'b0}};
        end else begin
            hist[1]      <= hist[0];
            hist[0]      <= i_data;
            r_hist_valid <= {r_hist_valid[KERNEL_SIZE-3:0], 1'b1};
            r_row_count  <= r_row_count + 1;
        end
    end
end

always @(posedge clk) begin
    if (!rst_n) begin
        o_data  <= {ACC_WIDTH{1'b0}};
        done    <= 1'b0;
        o_valid <= 1'b0;
    end else if (start) begin
        o_data  <= {ACC_WIDTH{1'b0}};
        done    <= 1'b0;
        o_valid <= 1'b0;
    end else if (i_valid && r_hist_valid == {(KERNEL_SIZE-1){1'b1}}) begin
        o_data  <= i_data  * w[0]
                 + hist[0] * w[1]
                 + hist[1] * w[2];
        // new_sample-qualified so conv.v's column counter (gated on this
        // pulse) advances once per real row, not once per replica.
        done    <= (r_row_count == i_rows - 2) && new_sample;
        o_valid <= 1'b1;
    end else begin
        done    <= 1'b0;
        o_valid <= 1'b0;
    end
end

// Per-PE waveform dump, opt-in via -DPE_WAVE. Must stay off by default: this
// runs once per pe instance, and an unguarded $dumpvars here beats the
// enclosing testbench's $dumpfile to the punch, so Icarus opens its default
// dump.vcd and every later $dumpfile -- including tb.v's tb.fst -- is ignored.
// $dumpfile also has to come FIRST; the other order is what caused that.
`ifdef PE_WAVE
initial begin
    $dumpfile("pe_wave.vcd");
    $dumpvars(0, pe);
end
`endif

endmodule
