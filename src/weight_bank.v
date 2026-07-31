// =============================================================================
//  weight_bank.v — one conv instance's kernel weight registers
// =============================================================================
//
//  Holds the KERNEL_SIZE*KERNEL_SIZE weight taps for a single conv instance.
//  wr_tap selects which tap register within this bank to write.
//
// =============================================================================

module weight_bank #(
    parameter DATA_WIDTH  = 8,
    parameter KERNEL_SIZE = 3
)(
    input  wire                                         clk,
    input  wire                                         rst_n,

    input  wire                                         wr_en,
    input  wire [$clog2(KERNEL_SIZE*KERNEL_SIZE)-1:0]    wr_tap,
    input  wire [DATA_WIDTH-1:0]                         wr_data,

    output wire [KERNEL_SIZE*KERNEL_SIZE*DATA_WIDTH-1:0] weights_out
);

localparam N_TAPS = KERNEL_SIZE * KERNEL_SIZE;

reg [DATA_WIDTH-1:0] r_weights [0:N_TAPS-1];

genvar gt;
generate
    for (gt = 0; gt < N_TAPS; gt = gt + 1) begin : g_tap_out
        assign weights_out[gt*DATA_WIDTH +: DATA_WIDTH] = r_weights[gt];
    end
endgenerate

integer t;
always @(posedge clk) begin
    if (!rst_n) begin
        for (t = 0; t < N_TAPS; t = t + 1) r_weights[t] <= 0;
    end else if (wr_en) begin
        r_weights[wr_tap] <= wr_data;
    end
end

endmodule
