#ifndef IMGPROC_GSC_COMMON_HPP
#define IMGPROC_GSC_COMMON_HPP

#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"

// Common AXI4-Stream video packet types shared across GSC blocks.
// 16-bit pixel payload with 1-bit USER/ID/DEST sidebands.
typedef ap_axiu<16, 1, 1, 1> packet_t;
typedef hls::stream<packet_t> packet_t_stream;

#endif // IMGPROC_GSC_COMMON_HPP
