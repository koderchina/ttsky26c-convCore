module relu #(
 parameter DATA_WIDTH = 16,
 parameter INPUT_WIDTH = 100,
 parameter INPUT_HEIGHT = 8
)(
    input wire                          i_valid,
    input wire signed [DATA_WIDTH-1:0]  i_data,
    input wire [$clog2(INPUT_HEIGHT)-1:0]          i_row,
    input wire [$clog2(INPUT_WIDTH)-1:0]          i_col,
    output wire                         o_valid,
    output reg signed [DATA_WIDTH-1:0]  o_data,
    output wire [$clog2(INPUT_HEIGHT)-1:0]         o_row,
    output wire [$clog2(INPUT_WIDTH)-1:0]         o_col
);

assign o_row = i_row;
assign o_col = i_col;
assign o_valid = i_valid;

always @(*) begin
    if (i_data <= 0) begin 
        o_data = 0;
    end else begin
        o_data = i_data;
    end
end
endmodule