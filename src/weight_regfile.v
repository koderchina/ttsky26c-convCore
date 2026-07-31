// =============================================================================
//  weight_regfile.v  —  weight/bias register file
// =============================================================================
//
//  Holds ONE output channel's parameters: the K*K kernel for each input
//  channel, plus that channel's single bias. An output channel is an
//  independent full convolution, so the chip computes one per pass and the
//  host re-programs this file before streaming the image again for the next.
//  That is why there is no (ic,oc) composite index any more -- the only axis
//  left is the input channel.
//
//  Address map (one 4-bit register per index, addresses are NIBBLE addresses
//  measured from this file's own base -- convCore subtracts the config
//  region's size before handing the bus over):
//
//    Weights:
//      addr = ic * KERNEL_SIZE * KERNEL_SIZE + kw * KERNEL_SIZE + tap
//        ic   = 0 .. IN_CHANNELS-1        (input channel)
//        kw   = 0 .. KERNEL_SIZE-1        (PE / kernel column)
//        tap  = 0 .. KERNEL_SIZE-1        (0 = newest sample weight)
//
//    Bias:
//      addr = MAX_IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE   (a single register)
//
//  The weight region boundary is COMPILE-TIME, derived from the MAX_* envelope
//  rather than from the runtime IN_CHANNELS. That keeps the write decode free
//  of runtime multipliers; the cost is that the host always writes a
//  fixed-size register image and simply skips over unused channel slots.
//
//  weights_out layout mirrors the address map so convLayer can slice directly:
//    weights_out[(ic*K*K + kw*K + tap)*DATA_WIDTH +: DATA_WIDTH]
//
//  Storage is banked one weight_bank per input channel (KERNEL_SIZE*KERNEL_SIZE
//  registers each) rather than one flat array, keeping every generate-for trip
//  count at MAX_IN_CHANNELS instead of MAX_IN_CHANNELS*K*K.
//
// =============================================================================

module weight_regfile #(
    parameter DATA_WIDTH       = 4,    // nibble width — one register per address
    parameter KERNEL_SIZE      = 3,
    parameter MAX_IN_CHANNELS  = 4,
    parameter REG_ADDR_W       = 9
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // Register write port — single-cycle, no handshake needed.
    input  wire                          wr_en,
    input  wire [REG_ADDR_W-1:0]         wr_addr,
    input  wire [DATA_WIDTH-1:0]         wr_data,

    // Weight/bias output buses (combinatorial reads)
    output wire [MAX_IN_CHANNELS*KERNEL_SIZE*KERNEL_SIZE*DATA_WIDTH-1:0] weights_out,
    output wire [DATA_WIDTH-1:0]                                         biases_out
);

localparam TAPS_PER_IC   = KERNEL_SIZE * KERNEL_SIZE;
localparam N_WEIGHT_REGS = MAX_IN_CHANNELS * TAPS_PER_IC;   // bias sits at this address
localparam TAP_ADDR_W    = $clog2(TAPS_PER_IC);

// ---- Weight storage — one bank per input channel ---------------------------
wire                    wr_is_weight = wr_en && (wr_addr < N_WEIGHT_REGS);
wire [REG_ADDR_W-1:0]   ic_idx       = wr_addr / TAPS_PER_IC;
wire [TAP_ADDR_W-1:0]   tap_idx      = wr_addr % TAPS_PER_IC;

genvar gi;
generate
    for (gi = 0; gi < MAX_IN_CHANNELS; gi = gi + 1) begin : g_bank
        weight_bank #(
            .DATA_WIDTH  (DATA_WIDTH),
            .KERNEL_SIZE (KERNEL_SIZE)
        ) u_bank (
            .clk         (clk),
            .rst_n       (rst_n),
            .wr_en       (wr_is_weight && (ic_idx == gi)),
            .wr_tap      (tap_idx),
            .wr_data     (wr_data),
            .weights_out (weights_out[gi*TAPS_PER_IC*DATA_WIDTH +: TAPS_PER_IC*DATA_WIDTH])
        );
    end
endgenerate

// ---- Bias storage — one register, for the single resident output channel ----
reg signed [DATA_WIDTH-1:0] r_bias;

assign biases_out = r_bias;

always @(posedge clk) begin
    if (!rst_n)
        r_bias <= 0;
    else if (wr_en && wr_addr == N_WEIGHT_REGS)
        r_bias <= wr_data;
end

endmodule
