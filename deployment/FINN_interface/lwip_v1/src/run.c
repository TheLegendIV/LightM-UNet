/*
 * Copyright (C) 2009 - 2019 Xilinx, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

#include <stdio.h>
#include <string.h>
#include <xparameters.h>
#include <stdbool.h>
#include "xil_cache.h"
#include "xil_io.h"
#include "xil_types.h"
#include "xstatus.h"
/* See hls_kernel_template.h for a generic, anonymized reference of the
 * AXI-Lite HLS kernel control pattern previously used by this file's
 * proprietary image-correction pipeline (now removed -- this platform's
 * hardware is a FINN-generated dataflow accelerator driven via idma0/odma0
 * register pokes, not AXI-Lite HLS kernels). Not included here since it is
 * unused boilerplate, kept only for future reference.
 */

/* AArch64 generic timer helpers (no BSP header needed) */
static inline uint64_t timer_get_count(void) {
    uint64_t v;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}
static inline uint64_t timer_get_freq_hz(void) {
    uint64_t f;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f));
    return f;
}
#include "lwip/err.h"
#include "lwip/tcp.h"
#if defined (__arm__) || defined (__aarch64__)
#include "xil_printf.h"
#endif
#ifndef VERBOSE_DEFAULT
/* Verbosity levels:
 * 1 = all, 2 = overview + state transitions, 3 = overview only.
 */
#define VERBOSE_DEFAULT 2
#endif

static int g_log_level = 3;

/* Logging levels:
 *   L1: all (including per RX/TX chunk chatter)
 *   L2: overview messages + state transitions
 *   L3: overview diagnostics only (sent image, received image, DMA copies, etc.)
 */
#define LOG_AT(level, ...) do { if ((level) >= g_log_level) xil_printf(__VA_ARGS__); } while (0)
#define LOG_L1(...) LOG_AT(1, __VA_ARGS__)
#define LOG_L2(...) LOG_AT(2, __VA_ARGS__)
#define LOG_L3(...) LOG_AT(3, __VA_ARGS__)
#define LOG_ERR(...) do { xil_printf(__VA_ARGS__); } while (0)

/* Default category for existing prints. */
#define DEBUG_PRINT(...) LOG_L2(__VA_ARGS__)

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/*
 * Runtime behavior / transfer sizing
 *
 * - TX: how many payload bytes we try to enqueue per send (not counting the
 *   4-byte length prefix). This is further clamped by tcp_sndbuf() at runtime.
 * - RX: maximum chunk length we accept from the client (protocol sanity).
 *
 * Override at build time via e.g. -DTX_CHUNK_BYTES=16384 -DRX_CHUNK_MAX_BYTES=65536
 */
#ifndef TX_CHUNK_BYTES
#define TX_CHUNK_BYTES (16U * 1024U)
#endif

#ifndef RX_CHUNK_MAX_BYTES
#define RX_CHUNK_MAX_BYTES (64U * 1024U)
#endif

/* DMA transfer timing: blocking wait + duration log */
#ifndef MONITOR_DMA
#define MONITOR_DMA 1
#endif

/* LwIP RX pixel timing: measure from first to last pixel byte received */
#ifndef MONITOR_LWIP_RX
#define MONITOR_LWIP_RX 1
#endif

/* FINN accelerator I/O contract (deployment/drivers/S12_dense: input
 * partition's driver.py -- idt=UINT8 ishape_packed=(1,64,64,1,1); output
 * partition's driver.py -- odt=INT20 oshape_packed=(1,64,64,5,3). The two
 * partitions are merged into one bitstream with a single idma0/odma0 pair;
 * the intermediate (1,32,32,4) INT9 tensor stays on-chip, never DMA'd.
 * Fixed at build time -- the accelerator only ever sees one 64x64 tile at a
 * time; the network's real 512x512 input is tiled/re-assembled in software
 * (see run_accel_all_tiles()), the accelerator has no notion of this.
 */
#define TILE_H (64U)
#define TILE_W (64U)
#define IMAGE_H (512U)
#define IMAGE_W (512U)
#define TILES_PER_ROW (IMAGE_W / TILE_W)                  /* 8 */
#define TILES_PER_COL (IMAGE_H / TILE_H)                  /* 8 */
#define NUM_TILES (TILES_PER_ROW * TILES_PER_COL)         /* 64 */

#define ACCEL_TILE_INPUT_BYTES  (TILE_H * TILE_W * 1U)    /* 4096 */
#define ACCEL_TILE_OUT_CHANNELS (5U)
#define ACCEL_TILE_OUT_BYTES_PER_ELEM (3U)                /* INT20, padded to 3 bytes */
#define ACCEL_TILE_OUT_BITWIDTH (20U)
#define ACCEL_TILE_OUTPUT_BYTES (TILE_H * TILE_W * ACCEL_TILE_OUT_CHANNELS * ACCEL_TILE_OUT_BYTES_PER_ELEM) /* 61440 */

/* Wire contract with the PC: one full image in, one combined per-pixel
 * argmax class-id map out (1 byte/pixel, nnU-Net convention: class index
 * 0..NUM_CLASSES-1) -- picking the winning class on-device makes the
 * accelerator's raw per-channel logits unnecessary to transmit. */
#define NETWORK_INPUT_BYTES  (IMAGE_H * IMAGE_W * 1U)     /* 262144 */
#define NETWORK_OUTPUT_BYTES (IMAGE_H * IMAGE_W * 1U)     /* 262144 */

#define MAX_IMAGE_SIZE ((NETWORK_INPUT_BYTES > NETWORK_OUTPUT_BYTES) ? NETWORK_INPUT_BYTES : NETWORK_OUTPUT_BYTES)

/* Set to 1 for smoke/echo testing with no bitstream loaded (no idma0/odma0
 * IODMA cores present, so XPAR_IDMA0/ODMA0_BASEADDR aren't even defined) --
 * RUN_ACCEL then just copies pixel_buf_src into pixel_buf_dst, truncated/
 * zero-padded to NETWORK_OUTPUT_BYTES, instead of driving the DMA registers
 * and tiling/argmax'ing. */
#ifndef ACCEL_LOOPBACK
#define ACCEL_LOOPBACK 0
#endif

/* Batch wire header, matches deployment/image_transfer_interface's
 * utilities/batch_img_header.py exactly: magic(4s "IMG3") +
 * payload_length(u32 BE, fixed per-image size) + width(u32 BE) +
 * height(u32 BE), all big-endian. Sent exactly ONCE per session/connection.
 * The client owns the image count -- there is no num_images field; instead
 * every image is preceded by a 1-byte marker (see IMG_MARKER_* below) and
 * the client sends IMG_MARKER_STOP when done. */
#define BATCH_HDR_SIZE_BYTES (16U)
#define BATCH_HDR_MAGIC "IMG3"

/* Batch-header ACK: "ACK" (3 bytes) + expected_output_bytes (u32 BE), so the
 * client learns the fixed per-image output size instead of assuming it. */
#define BATCH_ACK_SIZE_BYTES (7U)

/* Per-image marker byte, sent by the client before each unit following the
 * header ACK. No heartbeat -- just image-or-stop. */
#define IMG_MARKER_STOP  (0x00U)
#define IMG_MARKER_IMAGE (0x01U)

/* Generic 3-byte ack, reused for the header ack (as a prefix), per-image
 * completion, and the final stop ack -- the protocol is strictly lockstep
 * (client always waits for the specific reply to what it just sent), so a
 * single token is unambiguous. */
#define UNIT_ACK "ACK"
#define UNIT_ACK_SIZE_BYTES (3U)

/* idma0 / odma0 are FINN "IODMA" AXI-Lite (ap_ctrl_hs) cores, not classic
 * Xilinx axi_dma peripherals -- confirmed via the generated RTL wrapper
 * (m_axi_gmem0 + m_axis_0 + s_axi_control_0) and the reference PYNQ driver
 * (deployment/finn/driver_base.py execute_on_buffers/wait_until_finished),
 * which pokes exactly these three registers:
 *   0x00 control: bit0=ap_start(write) bit1=ap_done(read) bit2=ap_idle(read)
 *   0x10 dma_base: 32-bit pointer to the PS DDR buffer
 *   0x1C numReps: batch size (always 1 here)
 */
#define ACCEL_DMA_REG_CTRL    (0x00U)
#define ACCEL_DMA_REG_POINTER (0x10U)
#define ACCEL_DMA_REG_NUMREPS (0x1CU)
#define ACCEL_DMA_CTRL_AP_START (0x1U)
#define ACCEL_DMA_CTRL_AP_DONE  (0x2U)
#define ACCEL_DMA_CTRL_AP_IDLE  (0x4U)

#if !ACCEL_LOOPBACK
#define IDMA0_BASEADDR XPAR_IDMA0_BASEADDR
#define ODMA0_BASEADDR XPAR_ODMA0_BASEADDR
#endif

typedef enum {
    WAIT_BATCH_HEADER,  // Combined header length + one-time batch header
    WAIT_MARKER,        // Waiting for the 1-byte IMG_MARKER_STOP/IMG_MARKER_IMAGE
    WAIT_PIXELS,        // Accumulating current image's fixed-size raw payload
    RUN_ACCEL,          // Drive idma0 (in) + odma0 (out) and wait for completion
    SEND_PIXELS         // Streaming current image's fixed-size result back
} conn_state_t;

typedef struct {
    conn_state_t state;
    uint32_t expected_header_len;
    uint32_t header_received;

    /* Fixed for the whole session (validated == NETWORK_INPUT_BYTES/OUTPUT_BYTES). */
    uint32_t expected_input_bytes;
    uint32_t expected_output_bytes;

    /* Per-image progress (reset for each image). */
    uint32_t input_bytes_received;
	uint32_t tx_offset;

    /* Free-running count, cosmetic/logging only -- the client owns the
     * actual image count via IMG_MARKER_STOP, not the server. */
    uint32_t images_done;

	bool dma_started;
	bool dma_done;

    /* One-time batch header bytes. */
    u8 header_buf[BATCH_HDR_SIZE_BYTES];

    u8 *pixel_buf_src;     // Allocate dynamically after header
	u8 *pixel_buf_dst;     // Allocate dynamically after header

#if MONITOR_LWIP_RX
    uint64_t rx_t_start; /* cntvct_el0 captured on first pixel byte of current image */
#endif
} conn_ctx_t;

__attribute__((aligned(64))) static u8 pixel_buf_src[MAX_IMAGE_SIZE];
__attribute__((aligned(64))) static u8 pixel_buf_dst[MAX_IMAGE_SIZE];

/* Double-buffered per-tile DMA staging: while tile N's odma0 transfer is in
 * flight (writing into slot N%2), tile N-1's result (in the OTHER slot) is
 * being argmax'd on the CPU -- this is the "pipelining" between tiles. */
__attribute__((aligned(64))) static u8 tile_src_buf[2][ACCEL_TILE_INPUT_BYTES];
__attribute__((aligned(64))) static u8 tile_dst_buf[2][ACCEL_TILE_OUTPUT_BYTES];

static u32 g_tx_chunk_bytes = (u32)TX_CHUNK_BYTES;
static u32 g_rx_chunk_max_bytes = (u32)RX_CHUNK_MAX_BYTES;

#if !ACCEL_LOOPBACK
/* idma0 / odma0 register helpers (see ACCEL_DMA_REG_* above for the layout). */

static inline u32 accel_dma_read(UINTPTR base, u32 offset)
{
    return Xil_In32(base + offset);
}

static inline void accel_dma_write(UINTPTR base, u32 offset, u32 value)
{
    Xil_Out32(base + offset, value);
}

static inline int accel_dma_is_idle(UINTPTR base)
{
    return (accel_dma_read(base, ACCEL_DMA_REG_CTRL) & ACCEL_DMA_CTRL_AP_IDLE) != 0U;
}

static inline int accel_dma_is_done(UINTPTR base)
{
    return (accel_dma_read(base, ACCEL_DMA_REG_CTRL) & ACCEL_DMA_CTRL_AP_DONE) != 0U;
}

/* Program the buffer pointer + batch size (numReps=1) and launch. */
static inline void accel_dma_start(UINTPTR base, UINTPTR buf_addr)
{
    accel_dma_write(base, ACCEL_DMA_REG_POINTER, (u32)buf_addr);
    accel_dma_write(base, ACCEL_DMA_REG_NUMREPS, 1U);
    accel_dma_write(base, ACCEL_DMA_REG_CTRL, ACCEL_DMA_CTRL_AP_START);
}
#endif /* !ACCEL_LOOPBACK */

/* Decode one INT20 (little-endian, 3 packed bytes) accelerator output
 * element into a signed value -- see finn/util/data_packing.py's
 * reverse_endian=True convention (physical wire order is little-endian). */
static inline int32_t decode_int20_le(const u8 *p)
{
    u32 raw = (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16);
    raw &= (1U << ACCEL_TILE_OUT_BITWIDTH) - 1U;
    if (raw & (1U << (ACCEL_TILE_OUT_BITWIDTH - 1U))) {
        return (int32_t)raw - (int32_t)(1U << ACCEL_TILE_OUT_BITWIDTH);
    }
    return (int32_t)raw;
}

/* argmax(sigmoid(x)) == argmax(x) (sigmoid is monotonic), so the winning
 * class per pixel is found directly from the raw logits -- no float sigmoid
 * needed. Writes one class-id byte (0..ACCEL_TILE_OUT_CHANNELS-1) per pixel
 * into this tile's place in the full 512x512 output image. */
static void argmax_tile_into_output(conn_ctx_t *ctx, uint32_t tile_idx, const u8 *tile_raw)
{
    uint32_t tile_row = tile_idx / TILES_PER_ROW;
    uint32_t tile_col = tile_idx % TILES_PER_ROW;
    uint32_t row_off = tile_row * TILE_H;
    uint32_t col_off = tile_col * TILE_W;
    u8 class_row[TILE_W];

    for (uint32_t r = 0; r < TILE_H; r++) {
        for (uint32_t c = 0; c < TILE_W; c++) {
            uint32_t pix = r * TILE_W + c;
            const u8 *pix_base = tile_raw + (size_t)pix * ACCEL_TILE_OUT_CHANNELS * ACCEL_TILE_OUT_BYTES_PER_ELEM;
            int32_t best_val = decode_int20_le(pix_base);
            u8 best_ch = 0;
            for (uint32_t ch = 1; ch < ACCEL_TILE_OUT_CHANNELS; ch++) {
                int32_t v = decode_int20_le(pix_base + ch * ACCEL_TILE_OUT_BYTES_PER_ELEM);
                if (v > best_val) {
                    best_val = v;
                    best_ch = (u8)ch;
                }
            }
            class_row[c] = best_ch;
        }
        memcpy(&ctx->pixel_buf_dst[(row_off + r) * IMAGE_W + col_off], class_row, TILE_W);
    }
}

#if !ACCEL_LOOPBACK
/* Drive the accelerator over all NUM_TILES 64x64 tiles of the current
 * 512x512 image, row-major left-to-right/top-to-bottom, double-buffered so
 * tile N's CPU-side argmax overlaps tile N+1's hardware DMA time instead of
 * running strictly one tile at a time. */
static void run_accel_all_tiles(conn_ctx_t *ctx)
{
    int have_prev = 0;
    uint32_t prev_tile_idx = 0;
    int prev_slot = 0;

    for (uint32_t t = 0; t < NUM_TILES; t++) {
        int slot = (int)(t % 2U);
        uint32_t tile_row = t / TILES_PER_ROW;
        uint32_t tile_col = t % TILES_PER_ROW;
        uint32_t row_off = tile_row * TILE_H;
        uint32_t col_off = tile_col * TILE_W;

        for (uint32_t r = 0; r < TILE_H; r++) {
            memcpy(&tile_src_buf[slot][r * TILE_W],
                   &ctx->pixel_buf_src[(row_off + r) * IMAGE_W + col_off],
                   TILE_W);
        }

        Xil_DCacheFlushRange((UINTPTR)tile_src_buf[slot], ACCEL_TILE_INPUT_BYTES);
        Xil_DCacheInvalidateRange((UINTPTR)tile_dst_buf[slot], ACCEL_TILE_OUTPUT_BYTES);

        while (!accel_dma_is_idle((UINTPTR)ODMA0_BASEADDR)) { }

        /* Launch odma0 (drain) first, then idma0 (feed), matching the
         * reference PYNQ driver's ordering. */
        accel_dma_start((UINTPTR)ODMA0_BASEADDR, (UINTPTR)tile_dst_buf[slot]);
        accel_dma_start((UINTPTR)IDMA0_BASEADDR, (UINTPTR)tile_src_buf[slot]);

        /* Overlap: argmax the PREVIOUS tile's already-completed result while
         * THIS tile's DMA is in flight. */
        if (have_prev) {
            argmax_tile_into_output(ctx, prev_tile_idx, tile_dst_buf[prev_slot]);
        }

        while (!accel_dma_is_done((UINTPTR)ODMA0_BASEADDR)) { }
        Xil_DCacheInvalidateRange((UINTPTR)tile_dst_buf[slot], ACCEL_TILE_OUTPUT_BYTES);

        have_prev = 1;
        prev_tile_idx = t;
        prev_slot = slot;
    }
    /* Last tile has no successor to overlap with -- argmax it now. */
    if (have_prev) {
        argmax_tile_into_output(ctx, prev_tile_idx, tile_dst_buf[prev_slot]);
    }
}
#endif /* !ACCEL_LOOPBACK */

void set_verbosity(int v)
{
    /* Verbosity selection:
     * - v==0: quiet-ish (L3 only)
     * - v==1: full verbosity (L1)
     * - v==2/3: explicit level
     */
    if (v <= 0) {
        g_log_level = 3;
    } else if (v == 1) {
        g_log_level = 1;
    } else if (v == 2 || v == 3) {
        g_log_level = v;
    } else {
        g_log_level = 1;
    }
}

void set_chunk_sizes(u32 tx_chunk_bytes, u32 rx_chunk_max_bytes)
{
    if (tx_chunk_bytes != 0U) {
        g_tx_chunk_bytes = tx_chunk_bytes;
    }
    if (rx_chunk_max_bytes != 0U) {
        g_rx_chunk_max_bytes = rx_chunk_max_bytes;
    }
}

/* Batch wire header helpers, matching utilities/batch_img_header.py:
 * magic(4s "IMG3") + payload_length(u32 BE) + width(u32 BE) + height(u32 BE).
 * All fields are network byte order.
 */
typedef struct {
    u32 payload_length;
    u32 width;
    u32 height;
} batch_hdr_t;

static int parse_batch_header(const u8 *buf, u32 len, batch_hdr_t *out)
{
    u32 be_val;
    if (!buf || !out || len < BATCH_HDR_SIZE_BYTES) {
        return 0;
    }
    if (memcmp(buf, BATCH_HDR_MAGIC, 4) != 0) {
        return 0;
    }
    memcpy(&be_val, buf + 4, 4);  out->payload_length = ntohl(be_val);
    memcpy(&be_val, buf + 8, 4);  out->width           = ntohl(be_val);
    memcpy(&be_val, buf + 12, 4); out->height          = ntohl(be_val);
    return 1;
}

void print_app_header()
{
#if (LWIP_IPV6==0)
	LOG_L2("\n\r\n\r-----lwIP TCP server ------\n\r");
#else
    LOG_L2("\n\r\n\r-----lwIPv6 TCP server ------\n\r");
#endif
    LOG_L2("TCP image batches are streamed through the FINN accelerator (idma0/odma0), each result echoed back in turn\n\r");
}

/* lwIP helpers */

/* Kick off / continue streaming the current image's output back to the
 * client. Called once when SEND_PIXELS begins (poll_callback) and again on
 * every sent_callback until expected_output_bytes have gone out. No length
 * prefix is sent -- the client already knows the fixed per-image output size
 * from the one-time batch header. */
static void send_next_output_chunk(struct tcp_pcb *tpcb, conn_ctx_t *ctx)
{
    if (ctx->tx_offset >= ctx->expected_output_bytes) {
        return;
    }
    uint32_t remaining = ctx->expected_output_bytes - ctx->tx_offset;
    u32 sndbuf_payload = (u32)tcp_sndbuf(tpcb);
    uint32_t len_chunk = (uint32_t)MIN(remaining, MIN((uint32_t)g_tx_chunk_bytes, sndbuf_payload));
    if (len_chunk == 0U) {
        return;
    }
    if (tcp_write(tpcb, ctx->pixel_buf_dst + ctx->tx_offset, len_chunk, TCP_WRITE_FLAG_COPY) == ERR_OK) {
        tcp_output(tpcb);
        ctx->tx_offset += len_chunk;
    }
}

static err_t recv_callback(void *arg, struct tcp_pcb *tpcb, struct pbuf *p, err_t err) {
    LOG_L1("Receive callback invoked\r\n");
	conn_ctx_t *ctx = (conn_ctx_t *)arg;
	
	/* do not read the packet if we are not in ESTABLISHED state */
	if (!p) {
		tcp_close(tpcb);
		tcp_recv(tpcb, NULL);
		return ERR_OK;
	}

	// Per-pbuf chatter
    LOG_L1("Received pbuf: total_len=%d, len=%d\r\n", p->tot_len, p->len);

    struct pbuf *q = p;
    while (q) {
        u8 *data = (u8 *)q->payload;
		// length of current packet buffer (contains aribitrary number of packet payloads)
		uint16_t len = q->len;
        while (len > 0) {
            switch (ctx->state) {
            
            case WAIT_BATCH_HEADER: {
                DEBUG_PRINT("State: WAIT_BATCH_HEADER\r\n");

                /* Read (and discard) the 4-byte header length prefix -- kept on
                 * the wire for framing-format compatibility, but the header is
                 * now a fixed BATCH_HDR_SIZE_BYTES, so its value is unused. */
                if (ctx->expected_header_len == 0) {
                    if (len >= 4) {
                        data += 4; len -= 4;
                        ctx->expected_header_len = BATCH_HDR_SIZE_BYTES;
                    } else {
                        DEBUG_PRINT("Partial header length received!\r\n");
						len = 0;
                        break;
                    }
                }

                {
                    uint32_t copy_len = MIN(len, BATCH_HDR_SIZE_BYTES - ctx->header_received);
                    memcpy(ctx->header_buf + ctx->header_received, data, copy_len);
                    ctx->header_received += copy_len;
                    data += copy_len; len -= copy_len;

                    if (ctx->header_received == BATCH_HDR_SIZE_BYTES) {
                        /* Stage buffers in PS DDR. */
                        ctx->pixel_buf_src = pixel_buf_src;
                        ctx->pixel_buf_dst = pixel_buf_dst;

                        /* New session: clear all per-session/per-image state. */
                        ctx->dma_done = false;
                        ctx->dma_started = false;
                        ctx->tx_offset = 0;
                        ctx->input_bytes_received = 0;
                        ctx->images_done = 0;

                        batch_hdr_t hdr;
                        if (!parse_batch_header(ctx->header_buf, BATCH_HDR_SIZE_BYTES, &hdr)) {
                            LOG_ERR("ERROR: not a batch IMG3 header (bad magic/size)\r\n");
                            return ERR_OK;
                        }
                        if (hdr.payload_length != NETWORK_INPUT_BYTES) {
                            LOG_ERR("ERROR: payload_length=%u, server requires exactly %u bytes (512x512 image)\r\n",
                                    (unsigned)hdr.payload_length, (unsigned)NETWORK_INPUT_BYTES);
                            return ERR_OK;
                        }

                        ctx->expected_input_bytes = NETWORK_INPUT_BYTES;
                        ctx->expected_output_bytes = NETWORK_OUTPUT_BYTES;

                        LOG_L3("Received IMG3 header: %luB in / %luB out per image (w=%lu h=%lu)\r\n",
                               (unsigned long)ctx->expected_input_bytes,
                               (unsigned long)ctx->expected_output_bytes,
                               (unsigned long)hdr.width,
                               (unsigned long)hdr.height);

                        /* Header ACK: "ACK" + expected_output_bytes (u32 BE) so
                         * the client learns the fixed per-image output size. */
                        {
                            u8 ack_buf[BATCH_ACK_SIZE_BYTES];
                            u32 be_out_bytes = htonl(ctx->expected_output_bytes);
                            memcpy(ack_buf, UNIT_ACK, UNIT_ACK_SIZE_BYTES);
                            memcpy(ack_buf + UNIT_ACK_SIZE_BYTES, &be_out_bytes, 4);
                            tcp_write(tpcb, ack_buf, sizeof(ack_buf), TCP_WRITE_FLAG_COPY);
                            tcp_output(tpcb);
                        }
                        DEBUG_PRINT("ACK+output_size sent to PC (header)\r\n");
                        ctx->state = WAIT_MARKER;
                    }
                }
                break;
            }

            case WAIT_MARKER: {
                DEBUG_PRINT("State: WAIT_MARKER\r\n");
                u8 marker = data[0];
                data += 1; len -= 1;

                if (marker == IMG_MARKER_STOP) {
                    LOG_L3("STOP received (%u image(s) processed); session done\r\n",
                           (unsigned)ctx->images_done);
                    tcp_write(tpcb, UNIT_ACK, UNIT_ACK_SIZE_BYTES, TCP_WRITE_FLAG_COPY);
                    tcp_output(tpcb);

                    /* Ready for a new header on this same connection. */
                    ctx->state = WAIT_BATCH_HEADER;
                    ctx->expected_header_len = 0;
                    ctx->header_received = 0;
                    ctx->images_done = 0;
                    ctx->expected_input_bytes = 0;
                    ctx->expected_output_bytes = 0;
                    ctx->input_bytes_received = 0;
                    ctx->tx_offset = 0;
                    ctx->dma_started = false;
                    ctx->dma_done = false;
                    memset(ctx->header_buf, 0, sizeof(ctx->header_buf));
                } else if (marker == IMG_MARKER_IMAGE) {
                    ctx->dma_done = false;
                    ctx->dma_started = false;
                    ctx->tx_offset = 0;
                    ctx->input_bytes_received = 0;
                    ctx->state = WAIT_PIXELS;
                } else {
                    LOG_ERR("ERROR: unknown marker byte 0x%02x, aborting connection\r\n", (unsigned)marker);
                    tcp_abort(tpcb);
                    return ERR_ABRT;
                }
                break;
            }

            case WAIT_PIXELS: {
                if (ctx->input_bytes_received == 0) {
                    LOG_L2("State: WAIT_PIXELS (image %u, expecting %u bytes)\r\n",
                           (unsigned)(ctx->images_done + 1), ctx->expected_input_bytes);
                } else {
                    LOG_L1("State: WAIT_PIXELS\r\n");
                }

                /* No per-chunk framing: the image is exactly expected_input_bytes
                 * of raw payload, back-to-back, so just accumulate until full. */
                uint32_t to_copy = MIN(len, ctx->expected_input_bytes - ctx->input_bytes_received);
                if (to_copy > 0) {
                    memcpy(ctx->pixel_buf_src + ctx->input_bytes_received, data, to_copy);

                    #if MONITOR_LWIP_RX
                    if (ctx->input_bytes_received == 0) {
                        ctx->rx_t_start = timer_get_count();
                    }
                    #endif

                    ctx->input_bytes_received += to_copy;
                    data += to_copy; len -= to_copy;
                }

                if (ctx->input_bytes_received >= ctx->expected_input_bytes) {
                    ctx->dma_done = false;
                    ctx->dma_started = false;
                    ctx->tx_offset = 0;

                    ctx->state = RUN_ACCEL;
                    LOG_L3("Received image %u payload (%u bytes); starting pipeline\r\n",
                           (unsigned)(ctx->images_done + 1), ctx->expected_input_bytes);
                    #if MONITOR_LWIP_RX
                    {
                        uint64_t elapsed_ticks = timer_get_count() - ctx->rx_t_start;
                        uint64_t elapsed_us = elapsed_ticks * 1000000ULL / timer_get_freq_hz();
                        LOG_L3("LWIP RX: %u bytes in %llu us\r\n",
                               (unsigned)ctx->expected_input_bytes,
                               (unsigned long long)elapsed_us);
                    }
                    #endif
                }
                break;
            }

            default:
				// case state is handled in poll callback
                DEBUG_PRINT("State: other (%d), ignoring received data\r\n", ctx->state);
				len = 0;
                break;
            }
			
		}
		tcp_recved(tpcb, (u16_t)q->len);  // ACK all bytes in this pbuf
        q = q->next;
    }
    
    pbuf_free(p);                  // then free the chain once

    LOG_L1("Receive callback complete, current state: %d\r\n", ctx->state);
    return ERR_OK;
}

static err_t sent_callback(void *arg, struct tcp_pcb *tpcb, u16_t bytes_acked) {
    conn_ctx_t *ctx = (conn_ctx_t *)arg;
    LOG_L1("Sent callback, bytes_acked=%d\r\n", bytes_acked);
    if (ctx->state == SEND_PIXELS) {
        send_next_output_chunk(tpcb, ctx);

        if (ctx->tx_offset >= ctx->expected_output_bytes) {
            ctx->images_done += 1;
            LOG_L3("Image %u sent back (%u bytes)\r\n",
                   (unsigned)ctx->images_done, ctx->expected_output_bytes);

            ctx->dma_started = false;
            ctx->dma_done = false;
            ctx->tx_offset = 0;

            /* Per-image ACK after every image, then wait for the client's
             * next marker byte (IMG_MARKER_IMAGE or IMG_MARKER_STOP). */
            tcp_write(tpcb, UNIT_ACK, UNIT_ACK_SIZE_BYTES, TCP_WRITE_FLAG_COPY);
            tcp_output(tpcb);
            ctx->state = WAIT_MARKER;
        }
    }
    return ERR_OK;
}

static void reset_conn(struct tcp_pcb *tpcb, conn_ctx_t *ctx, int abort);

static err_t poll_callback(void *arg, struct tcp_pcb *tpcb)
{
    conn_ctx_t *ctx = (conn_ctx_t *)arg;

    switch (ctx->state) {
    case RUN_ACCEL: {
        DEBUG_PRINT("Poll: RUN_ACCEL\r\n");

        if (!ctx->pixel_buf_src || !ctx->pixel_buf_dst ||
            ctx->expected_input_bytes == 0 || ctx->expected_output_bytes == 0) {
            DEBUG_PRINT("RUN_ACCEL: invalid buffers/size\r\n");
            return ERR_OK;
        }

#if ACCEL_LOOPBACK
        /* No bitstream loaded -- echo the input straight back (truncated to
         * the class-map's 1 byte/pixel size) instead of driving idma0/odma0
         * and tiling/argmax'ing. Structural smoke test only. */
        {
            uint32_t copy_len = MIN(ctx->expected_input_bytes, ctx->expected_output_bytes);
            memcpy(ctx->pixel_buf_dst, ctx->pixel_buf_src, copy_len);
            if (ctx->expected_output_bytes > copy_len) {
                memset(ctx->pixel_buf_dst + copy_len, 0, ctx->expected_output_bytes - copy_len);
            }
            LOG_L3("RUN_ACCEL (loopback): echoed %u bytes, zero-padded to %u\r\n",
                   (unsigned)copy_len, (unsigned)ctx->expected_output_bytes);
        }
        ctx->dma_started = false;
        ctx->dma_done = true;
        ctx->tx_offset = 0;
        ctx->state = SEND_PIXELS;
        send_next_output_chunk(tpcb, ctx);
        return ERR_OK;
#else
        /* Whole image processed synchronously in one shot: NUM_TILES 64x64
         * tiles, each fed through idma0/odma0 and argmax'd into the combined
         * 512x512 class-id map, with tile N's argmax overlapping tile N+1's
         * DMA (see run_accel_all_tiles()). */
        ctx->dma_started = true;
        ctx->dma_done = false;

#if MONITOR_DMA
        {
            uint64_t t_start = timer_get_count();
            run_accel_all_tiles(ctx);
            uint64_t elapsed_ticks = timer_get_count() - t_start;
            uint64_t elapsed_us = elapsed_ticks * 1000000ULL / timer_get_freq_hz();
            LOG_L3("RUN_ACCEL done: %u tiles, image=%ux%u, time=%llu us\r\n",
                   (unsigned)NUM_TILES, (unsigned)IMAGE_W, (unsigned)IMAGE_H,
                   (unsigned long long)elapsed_us);
        }
#else
        run_accel_all_tiles(ctx);
#endif

        ctx->dma_started = false;
        ctx->dma_done = true;

        ctx->tx_offset = 0;
        ctx->state = SEND_PIXELS;
        send_next_output_chunk(tpcb, ctx);
        return ERR_OK;
#endif /* ACCEL_LOOPBACK */
    }

    default:
        // Other states idle in poll
        break;
    }

    return ERR_OK;
}

static void reset_conn(struct tcp_pcb *tpcb, conn_ctx_t *ctx, int abort)
{
    /* Reset connection state and close or abort the TCP control block. */

    /* Buffers are static in this app; do not free(). */
    ctx->pixel_buf_src = NULL;
    ctx->pixel_buf_dst = NULL;
    ctx->expected_header_len = 0;
    ctx->expected_input_bytes = 0;
    ctx->expected_output_bytes = 0;
    ctx->header_received = 0;

    ctx->images_done = 0;
    ctx->input_bytes_received = 0;
    ctx->tx_offset = 0;
    ctx->dma_started = false;
    ctx->dma_done = false;
    memset(ctx->header_buf, 0, sizeof(ctx->header_buf));
    ctx->state = WAIT_BATCH_HEADER;

    if (abort) tcp_abort(tpcb);
    else tcp_close(tpcb);
}

err_t accept_callback(void *arg, struct tcp_pcb *newpcb, err_t err) {
    conn_ctx_t *ctx = malloc(sizeof(conn_ctx_t));
    memset(ctx, 0, sizeof(conn_ctx_t));

	// Set initial state
    ctx->state = WAIT_BATCH_HEADER;
    ctx->expected_header_len = 0;
    ctx->expected_input_bytes = 0;
    ctx->expected_output_bytes = 0;
    ctx->header_received = 0;

    ctx->images_done = 0;
    ctx->input_bytes_received = 0;
	ctx->tx_offset = 0;
	ctx->dma_started = false;
	ctx->dma_done = false;

    ctx->pixel_buf_src = pixel_buf_src;
    ctx->pixel_buf_dst = pixel_buf_dst;

    tcp_arg(newpcb, ctx);
    tcp_recv(newpcb, recv_callback);
    tcp_sent(newpcb, sent_callback);
    tcp_poll(newpcb, poll_callback, 4);
    tcp_nagle_disable(newpcb);

    LOG_L2("Connection accepted\r\n");
    
	return ERR_OK;
}

int dma_init(void)
{
    /* idma0/odma0 are always-on FINN IODMA cores (see ACCEL_DMA_REG_* above) --
     * just confirm both are idle before the server starts accepting images. */
//    LOG_L2("idma0 ctrl=0x%08x odma0 ctrl=0x%08x\r\n",
//           (unsigned)accel_dma_read((UINTPTR)IDMA0_BASEADDR, ACCEL_DMA_REG_CTRL),
//           (unsigned)accel_dma_read((UINTPTR)ODMA0_BASEADDR, ACCEL_DMA_REG_CTRL));
//
//    if (!accel_dma_is_idle((UINTPTR)IDMA0_BASEADDR) || !accel_dma_is_idle((UINTPTR)ODMA0_BASEADDR)) {
//        LOG_ERR("WARN: idma0/odma0 not idle at startup\r\n");
//    }

    return XST_SUCCESS;
}

/* Main */

int start_application()
{
	struct tcp_pcb *pcb;
	err_t err;
	unsigned port = 7;

    dma_init();
    
	/* create new TCP PCB structure */
	pcb = tcp_new_ip_type(IPADDR_TYPE_ANY);
	if (!pcb) {
        DEBUG_PRINT("Error creating PCB. Out of Memory\n\r");
		return -1;
	}

	/* bind to specified @port */
	err = tcp_bind(pcb, IP_ANY_TYPE, port);
	if (err != ERR_OK) {
        DEBUG_PRINT("Unable to bind to port %d: err = %d\n\r", port, err);
		return -2;
	}

	/* we do not need any arguments to callback functions */
	tcp_arg(pcb, NULL);

	/* listen for connections */
	pcb = tcp_listen(pcb);
	if (!pcb) {
        DEBUG_PRINT("Out of memory while tcp_listen\n\r");
		return -3;
	}

    DEBUG_PRINT("Starting TCP server...\n\r");
	/* specify callback to use for incoming connections */
	tcp_accept(pcb, accept_callback);

    LOG_L2("TCP server started @ port %d\n\r", port);

	return 0;
}
