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
#include "xaxidma.h"
#include "xil_cache.h"
#include "xil_types.h"
#include "xstatus.h"
#include "xddr_read.h"
#include "xddr_write.h"
#include "xgrid_filter.h"
#include "xgrid_contrast.h"
#include "xscatter.h"
#include "xmedian_filter.h"
#include "xblock_passthrough.h"
#include "xcorrect.h"
#include "xaxipmon.h"
#include "gsc_utils.h"

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

/* Performance monitoring */
#ifndef MONITOR_PERFORMANCE
#define MONITOR_PERFORMANCE 1
#endif

/* DMA transfer timing: blocking wait + duration log */
#ifndef MONITOR_DMA
#define MONITOR_DMA 1
#endif

/* LwIP RX pixel timing: measure from first to last pixel byte received */
#ifndef MONITOR_LWIP_RX
#define MONITOR_LWIP_RX 1
#endif

/* GSC Constants */

#define ENABLE_BYPASS 0U // Set to 1 to skip kernel processing and just do DMA loopback (for testing)

#ifndef BYPASS_GFG
#define BYPASS_GFG ENABLE_BYPASS 
#endif

#ifndef BYPASS_GFP
#define BYPASS_GFP ENABLE_BYPASS
#endif

#ifndef BYPASS_GCG
#define BYPASS_GCG ENABLE_BYPASS
#endif

#ifndef BYPASS_GCP
#define BYPASS_GCP ENABLE_BYPASS
#endif

#ifndef BYPASS_SC
#define BYPASS_SC ENABLE_BYPASS
#endif

#ifndef BYPASS_MF
#define BYPASS_MF ENABLE_BYPASS
#endif

#ifndef BYPASS_COR
#define BYPASS_COR ENABLE_BYPASS
#endif

#ifndef IMAGE_WIDTH
#define IMAGE_WIDTH (1232U)
#endif

#ifndef IMAGE_HEIGHT
#define IMAGE_HEIGHT (904U)
#endif

#ifndef GRID_FILTER_Y_TAPS_FREQ_HZ
#define GRID_FILTER_Y_TAPS_FREQ_HZ (43.5f)
#endif

#ifndef GRID_FILTER_Y_TAPS_VERTICAL_BINNING_1BASED
#define GRID_FILTER_Y_TAPS_VERTICAL_BINNING_1BASED (2u)
#endif

#ifndef GRID_CONTRAST_GAUSS_SIGMA
#define GRID_CONTRAST_GAUSS_SIGMA (4.0f)
#endif

#ifndef GRID_CONTRAST_GAUSS_TRUNCATION
#define GRID_CONTRAST_GAUSS_TRUNCATION (3.0f)
#endif

#ifndef DEBUG_KERNELS
#define DEBUG_KERNELS 0
#endif

#define GC_TAP_FRAC_BITS   (14)
#define GC_GSC_Q_FRAC_BITS (14)
#define GC_TAP_SCALE       (1 << GC_TAP_FRAC_BITS)
#define GC_TAP_COUNT       (25)

#define GRID_TAP_FRAC_BITS (14)
#define GRID_TAP_SCALE     (1 << GRID_TAP_FRAC_BITS)
#define GRID_X_TAP_COUNT   (21)
#define GRID_Y_TAP_COUNT   (35)

#define MAX_IMAGE_SIZE (16U * 1024U * 1024U) // 16 MB is also maximum of DMA single mode transfer (24-bit length field)
/* FXD/FDXD image header support */
#define FXD_HEADER_FIXED_BYTES (128U)
#define FXD_MAGIC_OLD (0x46445844U) /* 'F''D''X''D' */
#define FXD_MAGIC_NEW (0x66645864U) /* 'f''d''X''d' */
#define FXD_TYPE_FLOAT32 (0)
#define FXD_TYPE_INT16   (1)
#define FXD_TYPE_UCHAR   (2)

/* HLS ddr_read/ddr_write constraints */
#ifndef HLS_DDR_MAX_WIDTH
#define HLS_DDR_MAX_WIDTH  (2048U)
#endif

#ifndef HLS_DDR_MAX_HEIGHT
#define HLS_DDR_MAX_HEIGHT (2048U)
#endif

/* PL DDR data mappings */
#ifndef MIG_DDR_BASEADDR
#define MIG_DDR_BASEADDR (0x000400000000ULL)
#endif

/* PL DDR mappings for HLS kernels / sources
 * - gain  at 0x000401000000
 * - ptnt  at 0x000402000000
 * - sink  at 0x000403000000
 */
#define PL_DDR_GAIN_BASE ((UINTPTR)(MIG_DDR_BASEADDR + 0x00000000ULL)) /* 0x000401000000 */
#define PL_DDR_PTNT_BASE ((UINTPTR)(MIG_DDR_BASEADDR + 0x01000000ULL)) /* 0x000402000000 */
#define PL_DDR_SINK_BASE ((UINTPTR)(MIG_DDR_BASEADDR + 0x02000000ULL)) /* 0x000403000000 */

/* IP base addresses */
#define DDR_READ_GAIN_BASEADDR    XPAR_DDR_READ_GAIN_BASEADDR
#define DDR_READ_PTNT_0_BASEADDR XPAR_DDR_READ_PTNT_0_BASEADDR
#define DDR_WRITE_BASEADDR         XPAR_DDR_WRITE_BASEADDR
#define PASSTHROUGH_BASEADDR       XPAR_BLOCK_PASSTHROUGH_0_BASEADDR
#define GRID_FILTER_BASEADDR XPAR_GRID_FILTER_GAIN_BASEADDR
#define GRID_FILTER_PTNT_BASEADDR XPAR_GRID_FILTER_PTNT_BASEADDR
#define GRID_CONTRAST_BASEADDR XPAR_GRID_CONTRAST_GAIN_BASEADDR
#define GRID_CONTRAST_PTNT_BASEADDR XPAR_GRID_CONTRAST_PTNT_BASEADDR
#define SCATTER_BASEADDR XPAR_SCATTER_0_BASEADDR
#define MEDIAN_FILTER_BASEADDR XPAR_MEDIAN_FILTER_0_BASEADDR
#define CORRECT_BASEADDR XPAR_CORRECT_0_BASEADDR
#define PASSTHROUGH_GFP_BASEADDR   XPAR_BLOCK_PASSTHROUGH_GFP_BASEADDR
#define DDR_WRITE_GFP_BASEADDR     XPAR_DDR_WRITE_GFP_BASEADDR

/* AXI Performance Monitor (bring canonical base into this module's scope) */
#define AXI_PERFMON_BASEADDR XPAR_XAXIPMON_0_BASEADDR

/* DMA constants */
#define DMA_DEV_ID 0
#define DMA_BASE XPAR_AXI_DMA_0_BASEADDR
#define DMA_MM2S_DMASR_OFFSET (0x04U)
#define DMA_S2MM_DMASR_OFFSET (0x34U)
#define DMA_BUF_SIZE 2048  // Enough for MTU or your packet size

/* Tap packing / quantization constants */

typedef enum {
    WAIT_HEADER,      // Combined header length + header
    WAIT_PIXELS,
    SEND_TO_PL,
    PROCESS,
    RECV_FROM_PL,
    SEND_HEADER,
	WAIT_ACK,
	SEND_PIXELS
} conn_state_t;

typedef struct {
    conn_state_t state;
    uint32_t expected_header_len;
    /* Input bytes received over TCP (raw image payload from client). */
    uint32_t expected_input_bytes;
    uint32_t expected_input_proc_bytes;
    uint32_t input_bytes_received;

    /* Output bytes that will be DMA'd back and sent to client. */
    uint32_t expected_output_bytes;
    uint32_t header_received;
	uint32_t tx_offset;
	bool ack_received;
	bool dma_started;
	bool dma_done;
	uint32_t pixel_chunk_remaining; // Remaining bytes in current pixel chunk
	uint32_t pixel_chunk_len;       // Current pixel chunk length

    /* Transported FXD header bytes (fixed header + optional comment). */
    u8 header_buf[1024];
    char fxd_endian;
    u32 fxd_type_code_in;
    u32 fxd_frames;
    u32 fxd_commentlen;

    u8 *pixel_buf_src;     // Allocate dynamically after header
	u8 *pixel_buf_dst;     // Allocate dynamically after header

    /* Image geometry for kernel programming */
    u32 img_width;
    u32 img_height;

    /* Processing state machine (driven by tcp_poll):
     * 0: not started
     * 1: DMA PS->PL (to PL_DDR_IN_BASE)
     * 2: kernels running (DDR write/read)
     * 3: DMA PL->PS (from PL_DDR_OUT_BASE)
     */
    u8 proc_phase;

    /* Kernel init flag for this connection */
    bool kernels_ready;

    /* Kernel start tracking (per image transaction) */
    bool kernels_started;

    /* ap_done is clear-on-read in HLS IP; latch completion per transaction. */
    bool ddr_read_gain_done_seen;
    bool ddr_read_ptnt_0_done_seen;
    bool ddr_write_done_seen;
    bool grid_filter_gain_done_seen;
    bool grid_contrast_gain_done_seen;
    bool grid_filter_ptnt_done_seen;
    bool grid_contrast_ptnt_done_seen;
    bool scatter_done_seen;
    bool passthrough_done_seen;
    bool passthrough_gfp_done_seen;
    bool median_filter_done_seen;
    bool correct_done_seen;
    bool ddr_write_gfp_done_seen;
    /* DMA copy phase for loopback DMA memcpy:
     * 0: not started
     * 1: PS->MIG in progress
     * 2: MIG->PS in progress
     */
    u8 dma_phase;
    bool got_gain;

#if MONITOR_LWIP_RX
    uint64_t rx_t_start; /* cntvct_el0 captured on first pixel byte */
#endif
} conn_ctx_t;

__attribute__((aligned(64))) static u8 pixel_buf_src[MAX_IMAGE_SIZE];
__attribute__((aligned(64))) static u8 pixel_buf_dst[MAX_IMAGE_SIZE];

static XAxiDma AxiDma; // Initialized in main() or platform init
/* AXI Performance Monitor instance for hardware global clock counter */
static XAxiPmon g_axipmon;
static bool g_axipmon_ready = false;

static u32 g_tx_chunk_bytes = (u32)TX_CHUNK_BYTES;
static u32 g_rx_chunk_max_bytes = (u32)RX_CHUNK_MAX_BYTES;

/* Kernel instances */

static XDdr_read g_ddr_read_gain;
static XDdr_read g_ddr_read_ptnt_0;
static XDdr_write g_ddr_write;
static XDdr_write g_ddr_write_gfp;
static XGrid_filter g_grid_filter;
static XGrid_filter g_grid_filter_ptnt;
static XGrid_contrast g_grid_contrast;
static XGrid_contrast g_grid_contrast_ptnt;
static XScatter g_scatter;
static XBlock_passthrough g_passthrough;
static XBlock_passthrough g_passthrough_gfp;
static XMedian_filter g_median_filter;
static XCorrect g_correct;
static bool g_kernels_inited = false;

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

/* FXD helpers */

static u32 rd_u32_le(const u8 *p)
{
    return ((u32)p[0]) | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
}

static u32 rd_u32_be(const u8 *p)
{
    return ((u32)p[3]) | ((u32)p[2] << 8) | ((u32)p[1] << 16) | ((u32)p[0] << 24);
}

static void wr_u32_le(u8 *p, u32 v)
{
    p[0] = (u8)(v & 0xFFU);
    p[1] = (u8)((v >> 8) & 0xFFU);
    p[2] = (u8)((v >> 16) & 0xFFU);
    p[3] = (u8)((v >> 24) & 0xFFU);
}

static void wr_u32_be(u8 *p, u32 v)
{
    p[3] = (u8)(v & 0xFFU);
    p[2] = (u8)((v >> 8) & 0xFFU);
    p[1] = (u8)((v >> 16) & 0xFFU);
    p[0] = (u8)((v >> 24) & 0xFFU);
}

static int fxd_detect_endian(const u8 *hdr, u32 len, char *out_endian)
{
    if (!hdr || len < 4U || !out_endian) {
        return 0;
    }

    u32 m_le = rd_u32_le(hdr);
    if (m_le == FXD_MAGIC_NEW || m_le == FXD_MAGIC_OLD) {
        *out_endian = '<';
        return 1;
    }

    u32 m_be = rd_u32_be(hdr);
    if (m_be == FXD_MAGIC_NEW || m_be == FXD_MAGIC_OLD) {
        *out_endian = '>';
        return 1;
    }

    return 0;
}

static void fxd_set_type_code(u8 *hdr, char endian, u32 type_code)
{
    /* type_code is the 5th int32 in the fixed header:
     * magic, cols, rows, frames, type_code, nquant_lev, ...
     */
    u8 *p = hdr + 16;
    if (endian == '>') {
        wr_u32_be(p, type_code);
    } else {
        wr_u32_le(p, type_code);
    }
}

static void fxd_set_magic(u8 *hdr, char endian, u32 magic)
{
    u8 *p = hdr;
    if (endian == '>') {
        wr_u32_be(p, magic);
    } else {
        wr_u32_le(p, magic);
    }
}

static void fxd_set_rows(u8 *hdr, char endian, u32 rows)
{
    /* rows is the 3rd int32 in the fixed header:
     * magic, cols, rows, frames, ...
     */
    u8 *p = hdr + 8;
    if (endian == '>') {
        wr_u32_be(p, rows);
    } else {
        wr_u32_le(p, rows);
    }
}

static u32 fxd_rd_i32(const u8 *p, char endian)
{
    return (endian == '>') ? rd_u32_be(p) : rd_u32_le(p);
}

static u32 fxd_bytes_per_pixel(u32 type_code)
{
    switch (type_code) {
    case FXD_TYPE_FLOAT32:
        return 4U;
    case FXD_TYPE_INT16:
        return 2U;
    case FXD_TYPE_UCHAR:
        return 1U;
    default:
        return 0U;
    }
}

/* Kernel helpers */

static int init_ddr_read_min(XDdr_read *inst, UINTPTR baseaddr)
{
    int st = XDdr_read_Initialize(inst, baseaddr);
    if (st != XST_SUCCESS) {
        return XST_FAILURE;
    }
    XDdr_read_DisableAutoRestart(inst);
    return XST_SUCCESS;
}

static int init_ddr_write_min(XDdr_write *inst, UINTPTR baseaddr)
{
    int st = XDdr_write_Initialize(inst, baseaddr);
    if (st != XST_SUCCESS) {
        return XST_FAILURE;
    }
    XDdr_write_DisableAutoRestart(inst);
    return XST_SUCCESS;
}

static void scatter_set_bypass(XScatter *inst, u32 en)
{
    XScatter_Set_bypass(inst, en);
}

static u32 scatter_get_bypass(XScatter *inst)
{
    return XScatter_Get_bypass(inst);
}

static void median_filter_set_bypass(XMedian_filter *inst, u32 en)
{
    XMedian_filter_Set_bypass(inst, en);
}

static u32 median_filter_get_bypass(XMedian_filter *inst)
{
    return XMedian_filter_Get_bypass(inst);
}

static void correct_set_bypass(XCorrect *inst, u32 en)
{
    XCorrect_Set_bypass(inst, en);
}

static u32 correct_get_bypass(XCorrect *inst)
{
    return XCorrect_Get_bypass(inst);
}

static int init_scatter_min(XScatter *inst)
{
    int st = XScatter_Initialize(inst, SCATTER_BASEADDR);
    u32 sc_ctrl;
    if (st != XST_SUCCESS) {
        return XST_FAILURE;
    }

    sc_ctrl = XScatter_ReadReg(inst->Ctrl_BaseAddress, XSCATTER_CTRL_ADDR_AP_CTRL);
    LOG_L2("SCATTER INIT: ctrl(before cfg)=0x%08lx\r\n", (unsigned long)sc_ctrl);

    XScatter_DisableAutoRestart(inst);
    scatter_set_bypass(inst, BYPASS_SC);

    sc_ctrl = XScatter_ReadReg(inst->Ctrl_BaseAddress, XSCATTER_CTRL_ADDR_AP_CTRL);
    LOG_L2("SCATTER INIT: ctrl(after cfg)=0x%08lx bypass=%lu\r\n",
           (unsigned long)sc_ctrl,
           (unsigned long)scatter_get_bypass(inst));

    if (sc_ctrl == 0U) {
        LOG_ERR("SCATTER INIT WARN: AP_CTRL is 0x00000000 right after init/config\r\n");
    }

    return XST_SUCCESS;
}

static void median_filter_set_gauss_taps(XMedian_filter *inst)
{
    u16 taps[GC_TAP_COUNT];
    u32 packed[(GC_TAP_COUNT + 1) / 2];
    u32 i;
    int16_t gauss_q15[GC_TAP_COUNT];

    if (gsc_make_gauss_kernel_1d_q(GC_GSC_Q_FRAC_BITS,
                                   GRID_CONTRAST_GAUSS_SIGMA,
                                   GRID_CONTRAST_GAUSS_TRUNCATION,
                                   gauss_q15,
                                   GC_TAP_COUNT)) {
        for (i = 0; i < GC_TAP_COUNT; ++i) {
            taps[i] = (u16)(s16)gauss_q15[i];   // Q14 directly, same sign convention as HLS
        }
        LOG_L2("MEDIAN_FILTER TAPS: gaussian sigma=%d.%03d trunc=%d.%03d\r\n",
               (int)GRID_CONTRAST_GAUSS_SIGMA,
               (int)((GRID_CONTRAST_GAUSS_SIGMA - (float)((int)GRID_CONTRAST_GAUSS_SIGMA)) * 1000.0f),
               (int)GRID_CONTRAST_GAUSS_TRUNCATION,
               (int)((GRID_CONTRAST_GAUSS_TRUNCATION - (float)((int)GRID_CONTRAST_GAUSS_TRUNCATION)) * 1000.0f));
    } else {
        s32 base = (GC_TAP_SCALE / GC_TAP_COUNT);
        s32 rem = (GC_TAP_SCALE - (base * GC_TAP_COUNT));
        LOG_ERR("MEDIAN_FILTER WARN: gauss tap generation failed, using box taps\r\n");
        for (i = 0; i < GC_TAP_COUNT; ++i) {
            taps[i] = (u16)base;
            if ((s32)i < rem) taps[i] = (u16)(taps[i] + 1);
        }
    }

    memset(packed, 0, sizeof(packed));
    for (i = 0; i < GC_TAP_COUNT; ++i) {
        packed[i >> 1] |= ((u32)((u16)taps[i])) << ((i & 1U) ? 16U : 0U);
    }

    for (i = 0; i < (u32)((GC_TAP_COUNT + 1) / 2); ++i) {
        XMedian_filter_WriteReg(inst->Ctrl_BaseAddress,
                               XMEDIAN_FILTER_CTRL_ADDR_GAUSS_1D_PACKED_DATA + (i * 4U),
                               packed[i]);
    }
}

static int init_median_filter_min(XMedian_filter *inst)
{
    int st = XMedian_filter_Initialize(inst, MEDIAN_FILTER_BASEADDR);
    u32 mf_ctrl;
    if (st != XST_SUCCESS) {
        return XST_FAILURE;
    }

    mf_ctrl = XMedian_filter_ReadReg(inst->Ctrl_BaseAddress, XMEDIAN_FILTER_CTRL_ADDR_AP_CTRL);
    LOG_L2("MEDIAN_FILTER INIT: ctrl(before cfg)=0x%08lx\r\n", (unsigned long)mf_ctrl);

    XMedian_filter_DisableAutoRestart(inst);
    median_filter_set_bypass(inst, BYPASS_MF);
    median_filter_set_gauss_taps(inst);

    mf_ctrl = XMedian_filter_ReadReg(inst->Ctrl_BaseAddress, XMEDIAN_FILTER_CTRL_ADDR_AP_CTRL);
    LOG_L2("MEDIAN_FILTER INIT: ctrl(after cfg)=0x%08lx bypass=%lu\r\n",
           (unsigned long)mf_ctrl,
           (unsigned long)median_filter_get_bypass(inst));

    if (mf_ctrl == 0U) {
        LOG_ERR("MEDIAN_FILTER INIT WARN: AP_CTRL is 0x00000000 right after init/config\r\n");
    }

    return XST_SUCCESS;
}

static int init_correct_min(XCorrect *inst, UINTPTR baseaddr)
{
    int st = XCorrect_Initialize(inst, baseaddr);
    u32 cor_ctrl;
    if (st != XST_SUCCESS) {
        return XST_FAILURE;
    }

    cor_ctrl = XCorrect_ReadReg(inst->Ctrl_bus_BaseAddress, XCORRECT_CTRL_BUS_ADDR_AP_CTRL);
    LOG_L2("CORRECT INIT: ctrl(before cfg)=0x%08lx\r\n", (unsigned long)cor_ctrl);

    XCorrect_DisableAutoRestart(inst);
    correct_set_bypass(inst, BYPASS_COR);

    cor_ctrl = XCorrect_ReadReg(inst->Ctrl_bus_BaseAddress, XCORRECT_CTRL_BUS_ADDR_AP_CTRL);
    LOG_L2("CORRECT INIT: ctrl(after cfg)=0x%08lx bypass=%lu\r\n",
           (unsigned long)cor_ctrl,
           (unsigned long)correct_get_bypass(inst));

    if (cor_ctrl == 0U) {
        LOG_ERR("CORRECT INIT WARN: AP_CTRL is 0x00000000 right after init/config\r\n");
    }

    return XST_SUCCESS;
}

static int init_passthrough_min(XBlock_passthrough *inst, UINTPTR baseaddr)
{
    int st = XBlock_passthrough_Initialize(inst, baseaddr);
    u32 pt_ctrl;
    if (st != XST_SUCCESS) {
        return XST_FAILURE;
    }

    pt_ctrl = XBlock_passthrough_ReadReg(inst->Ctrl_bus_BaseAddress, XBLOCK_PASSTHROUGH_CTRL_BUS_ADDR_AP_CTRL);
    LOG_L2("PASSTHROUGH INIT: ctrl(before cfg)=0x%08lx\r\n", (unsigned long)pt_ctrl);

    XBlock_passthrough_DisableAutoRestart(inst);

    pt_ctrl = XBlock_passthrough_ReadReg(inst->Ctrl_bus_BaseAddress, XBLOCK_PASSTHROUGH_CTRL_BUS_ADDR_AP_CTRL);
    LOG_L2("PASSTHROUGH INIT: ctrl(after cfg)=0x%08lx\r\n", (unsigned long)pt_ctrl);

    if (pt_ctrl == 0U) {
        LOG_ERR("PASSTHROUGH INIT WARN: AP_CTRL is 0x00000000 right after init/config\r\n");
    }

    return XST_SUCCESS;
}

static void grid_contrast_set_bypass(XGrid_contrast *inst, u32 en)
{
    XGrid_contrast_Set_bypass(inst, en);
}

static u32 grid_contrast_get_bypass(XGrid_contrast *inst)
{
    return XGrid_contrast_Get_bypass(inst);
}

static void grid_contrast_set_gauss_taps(XGrid_contrast *inst)
{
    u16 taps[GC_TAP_COUNT];
    u32 packed[(GC_TAP_COUNT + 1) / 2];
    u32 i;
    int16_t gauss_q15[GC_TAP_COUNT];

    if (gsc_make_gauss_kernel_1d_q(GC_GSC_Q_FRAC_BITS,
                                   GRID_CONTRAST_GAUSS_SIGMA,
                                   GRID_CONTRAST_GAUSS_TRUNCATION,
                                   gauss_q15,
                                   GC_TAP_COUNT)) {
        for (i = 0; i < GC_TAP_COUNT; ++i) {
            taps[i] = (u16)(s16)gauss_q15[i];   // Q14 directly, same sign convention as HLS
        }
        LOG_L2("GRID_CONTRAST TAPS: gaussian sigma=%d.%03d trunc=%d.%03d\r\n",
               (int)GRID_CONTRAST_GAUSS_SIGMA,
               (int)((GRID_CONTRAST_GAUSS_SIGMA - (float)((int)GRID_CONTRAST_GAUSS_SIGMA)) * 1000.0f),
               (int)GRID_CONTRAST_GAUSS_TRUNCATION,
               (int)((GRID_CONTRAST_GAUSS_TRUNCATION - (float)((int)GRID_CONTRAST_GAUSS_TRUNCATION)) * 1000.0f));
    } else {
        s32 base = (GC_TAP_SCALE / GC_TAP_COUNT);
        s32 rem = (GC_TAP_SCALE - (base * GC_TAP_COUNT));
        LOG_ERR("GRID_CONTRAST WARN: gauss tap generation failed (sigma=%d.%03d trunc=%d.%03d), using box taps\r\n",
                (int)GRID_CONTRAST_GAUSS_SIGMA,
                (int)((GRID_CONTRAST_GAUSS_SIGMA - (float)((int)GRID_CONTRAST_GAUSS_SIGMA)) * 1000.0f),
                (int)GRID_CONTRAST_GAUSS_TRUNCATION,
                (int)((GRID_CONTRAST_GAUSS_TRUNCATION - (float)((int)GRID_CONTRAST_GAUSS_TRUNCATION)) * 1000.0f));
        for (i = 0; i < GC_TAP_COUNT; ++i) {
            taps[i] = (u16)base;
            if ((s32)i < rem) taps[i] = (u16)(taps[i] + 1);
        }
    }

    memset(packed, 0, sizeof(packed));
    for (i = 0; i < GC_TAP_COUNT; ++i) {
        packed[i >> 1] |= ((u32)((u16)taps[i])) << ((i & 1U) ? 16U : 0U);
    }

    for (i = 0; i < (u32)((GC_TAP_COUNT + 1) / 2); ++i) {
        XGrid_contrast_WriteReg(inst->Ctrl_BaseAddress,
                               XGRID_CONTRAST_CTRL_ADDR_GAUSS_1D_PACKED_DATA + (i * 4U),
                               packed[i]);
    }
}

static int init_grid_contrast_min(XGrid_contrast *inst, UINTPTR baseaddr, u32 bypass_en)
{
    int st = XGrid_contrast_Initialize(inst, baseaddr);
    u32 gs_ctrl;
    if (st != XST_SUCCESS) {
        return XST_FAILURE;
    }

    gs_ctrl = XGrid_contrast_ReadReg(inst->Ctrl_BaseAddress, XGRID_CONTRAST_CTRL_ADDR_AP_CTRL);
    LOG_L2("GRID_CONTRAST INIT: ctrl(before cfg)=0x%08lx\r\n", (unsigned long)gs_ctrl);

    XGrid_contrast_DisableAutoRestart(inst);
    grid_contrast_set_bypass(inst, bypass_en);
    grid_contrast_set_gauss_taps(inst);

    gs_ctrl = XGrid_contrast_ReadReg(inst->Ctrl_BaseAddress, XGRID_CONTRAST_CTRL_ADDR_AP_CTRL);
    LOG_L2("GRID_CONTRAST INIT: ctrl(after cfg)=0x%08lx bypass=%lu\r\n",
           (unsigned long)gs_ctrl,
           (unsigned long)grid_contrast_get_bypass(inst));

    if (gs_ctrl == 0U) {
        LOG_ERR("GRID_CONTRAST INIT WARN: AP_CTRL is 0x00000000 right after init/config\r\n");
    }
    return XST_SUCCESS;
}

static void grid_filter_set_bypass(XGrid_filter *inst, u32 en)
{
    XGrid_filter_Set_bypass(inst, en);
}

static u32 grid_filter_get_bypass(XGrid_filter *inst)
{
    return XGrid_filter_Get_bypass(inst);
}

static s16 sat_s16_from_int(s32 v)
{
    if (v > 32767) return (s16)32767;
    if (v < -32768) return (s16)-32768;
    return (s16)v;
}

static void balance_unit_sum_q_inplace_s16(s16 *taps, u32 n, u32 frac_bits)
{
    if (!taps || n == 0U || frac_bits > 15U) {
        return;
    }

    s32 target = (s32)(1U << frac_bits);
    s32 sum = 0;
    u32 i;
    for (i = 0U; i < n; ++i) {
        sum += (s32)taps[i];
    }

    {
        s32 delta = target - sum;
        s32 base = delta / (s32)n;
        s32 rem = delta - (base * (s32)n);
        s32 center = (s32)(n / 2U);

        for (i = 0U; i < n; ++i) {
            taps[i] = sat_s16_from_int((s32)taps[i] + base);
        }

        if (rem != 0) {
            s32 step = (rem > 0) ? 1 : -1;
            s32 absrem = (rem > 0) ? rem : -rem;

            if (absrem > 0 && center >= 0 && center < (s32)n) {
                taps[center] = sat_s16_from_int((s32)taps[center] + step);
                --absrem;
            }

            for (s32 k = 1; absrem > 0; ++k) {
                s32 li = center - k;
                s32 ri = center + k;
                if (li < 0 || ri >= (s32)n) {
                    break;
                }
                if (absrem >= 2) {
                    taps[li] = sat_s16_from_int((s32)taps[li] + step);
                    taps[ri] = sat_s16_from_int((s32)taps[ri] + step);
                    absrem -= 2;
                } else {
                    taps[center] = sat_s16_from_int((s32)taps[center] + step);
                    absrem = 0;
                }
            }
        }
    }
}

static void grid_filter_set_default_coeffs(XGrid_filter *inst)
{
    u16 i;
    s16 coeff_x[GRID_X_TAP_COUNT];
    s16 coeff_y[GRID_Y_TAP_COUNT];
    u32 packed_x[11];
    u32 packed_y[18];
    s32 x_base;
    s32 x_rem;

    memset(coeff_x, 0, sizeof(coeff_x));
    memset(coeff_y, 0, sizeof(coeff_y));

    /* X taps unchanged: box filter in Q14 with exact unit-sum remainder distribution. */
    x_base = (GRID_TAP_SCALE / GRID_X_TAP_COUNT);
    x_rem  = (GRID_TAP_SCALE - (x_base * GRID_X_TAP_COUNT));

    for (i = 0; i < GRID_X_TAP_COUNT; ++i) {
        coeff_x[i] = (s16)x_base;
        if ((s32)i < x_rem) {
            coeff_x[i] = (s16)(coeff_x[i] + 1);
        }
    }

    if (!gsc_get_grid_filter_y_taps_q(GRID_TAP_FRAC_BITS,
                          GRID_FILTER_Y_TAPS_FREQ_HZ,
                          GRID_FILTER_Y_TAPS_VERTICAL_BINNING_1BASED,
                                      coeff_y,
                                      GRID_Y_TAP_COUNT)) {
        LOG_ERR("GRID INIT ERR: failed to build Y taps (freq=%d.%03d, bin=%lu)\r\n",
            (int)GRID_FILTER_Y_TAPS_FREQ_HZ,
            (int)((GRID_FILTER_Y_TAPS_FREQ_HZ - (float)((int)GRID_FILTER_Y_TAPS_FREQ_HZ)) * 1000.0f),
            (unsigned long)GRID_FILTER_Y_TAPS_VERTICAL_BINNING_1BASED);
        for (i = 0; i < GRID_Y_TAP_COUNT; ++i) {
            coeff_y[i] = 0;
        }
        coeff_y[GRID_Y_TAP_COUNT / 2U] = (s16)GRID_TAP_SCALE;
    } else {
        /* Ensure exact unit-sum in Q domain after quantization. */
        balance_unit_sum_q_inplace_s16(coeff_y, GRID_Y_TAP_COUNT, GRID_TAP_FRAC_BITS);
    }

    LOG_L2("GRID TAPS CFG: freq=%d.%03d bin=%lu\r\n",
           (int)GRID_FILTER_Y_TAPS_FREQ_HZ,
           (int)((GRID_FILTER_Y_TAPS_FREQ_HZ - (float)((int)GRID_FILTER_Y_TAPS_FREQ_HZ)) * 1000.0f),
           (unsigned long)GRID_FILTER_Y_TAPS_VERTICAL_BINNING_1BASED);

    LOG_L2("GRID TAPS X[%u]:", (unsigned)GRID_X_TAP_COUNT);
    for (i = 0; i < GRID_X_TAP_COUNT; ++i) {
        LOG_L2(" %d", (int)coeff_x[i]);
    }
    LOG_L2("\r\n");

    LOG_L2("GRID TAPS Y[%u]:", (unsigned)GRID_Y_TAP_COUNT);
    for (i = 0; i < GRID_Y_TAP_COUNT; ++i) {
        LOG_L2(" %d", (int)coeff_y[i]);
    }
    LOG_L2("\r\n");

    memset(packed_x, 0, sizeof(packed_x));
    memset(packed_y, 0, sizeof(packed_y));

    for (i = 0; i < GRID_X_TAP_COUNT; ++i) {
        packed_x[i >> 1] |= ((u32)((u16)coeff_x[i])) << ((i & 1U) ? 16U : 0U);
    }
    for (i = 0; i < GRID_Y_TAP_COUNT; ++i) {
        packed_y[i >> 1] |= ((u32)((u16)coeff_y[i])) << ((i & 1U) ? 16U : 0U);
    }

    for (i = 0; i < 11U; ++i) {
        XGrid_filter_WriteReg(inst->Ctrl_BaseAddress,
                              XGRID_FILTER_CTRL_ADDR_COEFF_X_PACKED_DATA + (i * 4U),
                              packed_x[i]);
    }
    for (i = 0; i < 18U; ++i) {
        XGrid_filter_WriteReg(inst->Ctrl_BaseAddress,
                              XGRID_FILTER_CTRL_ADDR_COEFF_Y_PACKED_DATA + (i * 4U),
                              packed_y[i]);
    }
}

static int init_grid_filter_min(XGrid_filter *inst, UINTPTR baseaddr, u32 bypass_en)
{
    int st = XGrid_filter_Initialize(inst, baseaddr);
    u32 gf_ctrl;
    if (st != XST_SUCCESS) {
        return XST_FAILURE;
    }

    gf_ctrl = XGrid_filter_ReadReg(inst->Ctrl_BaseAddress, XGRID_FILTER_CTRL_ADDR_AP_CTRL);
    LOG_L2("GRID INIT: ctrl(before cfg)=0x%08lx\r\n", (unsigned long)gf_ctrl);

    XGrid_filter_DisableAutoRestart(inst);
    grid_filter_set_bypass(inst, bypass_en); // Configure bypass mode
    grid_filter_set_default_coeffs(inst);

    gf_ctrl = XGrid_filter_ReadReg(inst->Ctrl_BaseAddress, XGRID_FILTER_CTRL_ADDR_AP_CTRL);
        LOG_L2("GRID INIT: ctrl(after cfg)=0x%08lx bypass=%lu\r\n",
           (unsigned long)gf_ctrl,
            (unsigned long)grid_filter_get_bypass(inst));

    if (gf_ctrl == 0U) {
        LOG_ERR("GRID INIT WARN: AP_CTRL is 0x00000000 right after init/config\r\n");
    }
    return XST_SUCCESS;
}

static void log_kernel_ready_idle_done(const char *phase)
{
    u32 rd_gain_ready  = (g_ddr_read_gain.IsReady       == XIL_COMPONENT_IS_READY) ? 1U : 0U;
    u32 rd_ptnt0_ready = (g_ddr_read_ptnt_0.IsReady     == XIL_COMPONENT_IS_READY) ? 1U : 0U;
    u32 sc_ready       = (g_scatter.IsReady              == XIL_COMPONENT_IS_READY) ? 1U : 0U;
    u32 pt_ready       = (g_passthrough.IsReady          == XIL_COMPONENT_IS_READY) ? 1U : 0U;
    u32 mf_ready       = (g_median_filter.IsReady        == XIL_COMPONENT_IS_READY) ? 1U : 0U;
    u32 gf_ready       = (g_grid_filter.IsReady          == XIL_COMPONENT_IS_READY) ? 1U : 0U;
    u32 gs_ready       = (g_grid_contrast.IsReady        == XIL_COMPONENT_IS_READY) ? 1U : 0U;
    u32 cor_ready      = (g_correct.IsReady              == XIL_COMPONENT_IS_READY) ? 1U : 0U;
    u32 wg_ready       = (g_ddr_write.IsReady            == XIL_COMPONENT_IS_READY) ? 1U : 0U;
    u32 pt_gfp_ready   = (g_passthrough_gfp.IsReady      == XIL_COMPONENT_IS_READY) ? 1U : 0U;
    u32 dw_gfp_ready   = (g_ddr_write_gfp.IsReady        == XIL_COMPONENT_IS_READY) ? 1U : 0U;

    int rd_idle = -1, rd_done = -1;
    int sc_idle = -1, sc_done = -1;
    int pt_idle = -1, pt_done = -1;
    int mf_idle = -1, mf_done = -1;
    int gf_idle = -1, gf_done = -1;
    int gs_idle = -1, gs_done = -1;
    int cor_idle = -1, cor_done = -1;
    int wg_idle = -1, wg_done = -1;
    int pt_gfp_idle = -1, pt_gfp_done = -1;
    int dw_gfp_idle = -1, dw_gfp_done = -1;

    if (rd_gain_ready) {
        rd_idle = (int)XDdr_read_IsIdle(&g_ddr_read_gain);
        rd_done = (int)XDdr_read_IsDone(&g_ddr_read_gain);
    }
    if (sc_ready) {
        sc_idle = (int)XScatter_IsIdle(&g_scatter);
        sc_done = (int)XScatter_IsDone(&g_scatter);
    }
    if (pt_ready) {
        pt_idle = (int)XBlock_passthrough_IsIdle(&g_passthrough);
        pt_done = (int)XBlock_passthrough_IsDone(&g_passthrough);
    }
    if (mf_ready) {
        mf_idle = (int)XMedian_filter_IsIdle(&g_median_filter);
        mf_done = (int)XMedian_filter_IsDone(&g_median_filter);
    }
    if (gf_ready) {
        gf_idle = (int)XGrid_filter_IsIdle(&g_grid_filter);
        gf_done = (int)XGrid_filter_IsDone(&g_grid_filter);
    }
    if (gs_ready) {
        gs_idle = (int)XGrid_contrast_IsIdle(&g_grid_contrast);
        gs_done = (int)XGrid_contrast_IsDone(&g_grid_contrast);
    }
    if (cor_ready) {
        cor_idle = (int)XCorrect_IsIdle(&g_correct);
        cor_done = (int)XCorrect_IsDone(&g_correct);
    }
    if (wg_ready) {
        wg_idle = (int)XDdr_write_IsIdle(&g_ddr_write);
        wg_done = (int)XDdr_write_IsDone(&g_ddr_write);
    }
    if (pt_gfp_ready) {
        pt_gfp_idle = (int)XBlock_passthrough_IsIdle(&g_passthrough_gfp);
        pt_gfp_done = (int)XBlock_passthrough_IsDone(&g_passthrough_gfp);
    }
    if (dw_gfp_ready) {
        dw_gfp_idle = (int)XDdr_write_IsIdle(&g_ddr_write_gfp);
        dw_gfp_done = (int)XDdr_write_IsDone(&g_ddr_write_gfp);
    }
    LOG_L2("KINIT[%s] ddr_read_gain: ready=%lu idle=%d done=%d | ddr_read_ptnt_0: ready=%lu | scatter: ready=%lu idle=%d done=%d | passthrough: ready=%lu idle=%d done=%d | median_filter: ready=%lu idle=%d done=%d | grid_filter_gain: ready=%lu idle=%d done=%d | grid_filter_ptnt: ready=%lu | grid_contrast_gain: ready=%lu idle=%d done=%d | grid_contrast_ptnt: ready=%lu | correct: ready=%lu idle=%d done=%d | ddr_write: ready=%lu idle=%d done=%d | passthrough_gfp: ready=%lu idle=%d done=%d | ddr_write_gfp: ready=%lu idle=%d done=%d\r\n",
        phase,
        (unsigned long)rd_gain_ready, rd_idle, rd_done,
        (unsigned long)rd_ptnt0_ready,
        (unsigned long)sc_ready, sc_idle, sc_done,
        (unsigned long)pt_ready, pt_idle, pt_done,
        (unsigned long)mf_ready, mf_idle, mf_done,
        (unsigned long)gf_ready, gf_idle, gf_done,
        (unsigned long)g_grid_filter_ptnt.IsReady,
        (unsigned long)gs_ready, gs_idle, gs_done,
        (unsigned long)g_grid_contrast_ptnt.IsReady,
        (unsigned long)cor_ready, cor_idle, cor_done,
        (unsigned long)wg_ready, wg_idle, wg_done,
        (unsigned long)pt_gfp_ready, pt_gfp_idle, pt_gfp_done,
        (unsigned long)dw_gfp_ready, dw_gfp_idle, dw_gfp_done);
}

static int kernels_init_once(void)
{
    if (g_kernels_inited) {
        return XST_SUCCESS;
    }

    // log_kernel_ready_idle_done("before");

    if (init_ddr_read_min(&g_ddr_read_gain, (UINTPTR)DDR_READ_GAIN_BASEADDR) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (init_ddr_read_min(&g_ddr_read_ptnt_0, (UINTPTR)DDR_READ_PTNT_0_BASEADDR) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (init_grid_filter_min(&g_grid_filter, (UINTPTR)GRID_FILTER_BASEADDR, BYPASS_GFG) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (init_grid_filter_min(&g_grid_filter_ptnt, (UINTPTR)GRID_FILTER_PTNT_BASEADDR, BYPASS_GFP) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (init_grid_contrast_min(&g_grid_contrast, (UINTPTR)GRID_CONTRAST_BASEADDR, BYPASS_GCG) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (init_grid_contrast_min(&g_grid_contrast_ptnt, (UINTPTR)GRID_CONTRAST_PTNT_BASEADDR, BYPASS_GCP) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (init_scatter_min(&g_scatter) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (init_passthrough_min(&g_passthrough, (UINTPTR)PASSTHROUGH_BASEADDR) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (init_passthrough_min(&g_passthrough_gfp, (UINTPTR)PASSTHROUGH_GFP_BASEADDR) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (init_median_filter_min(&g_median_filter) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (init_correct_min(&g_correct, (UINTPTR)CORRECT_BASEADDR) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (init_ddr_write_min(&g_ddr_write, DDR_WRITE_BASEADDR) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    if (init_ddr_write_min(&g_ddr_write_gfp, DDR_WRITE_GFP_BASEADDR) != XST_SUCCESS) {
        return XST_FAILURE;
    }

    #ifdef MONITOR_PERFORMANCE
    {
        XAxiPmon_Config *cfg = XAxiPmon_LookupConfig(AXI_PERFMON_BASEADDR);
        if (cfg != NULL && XAxiPmon_CfgInitialize(&g_axipmon, cfg, cfg->BaseAddress) == XST_SUCCESS) {
            g_axipmon_ready = true;
            LOG_L2("AXI PerfMon initialized\r\n");
        } else {
            LOG_L2("AXI PerfMon init failed\r\n");
        }
    }
    #endif
    
    LOG_L2("KERNEL MAP: ddr_read_gain=0x%08lx ddr_read_ptnt_0=0x%08lx scatter_0=0x%08lx passthrough_0=0x%08lx passthrough_gfp=0x%08lx median_filter_0=0x%08lx correct_0=0x%08lx ddr_write=0x%08lx ddr_write_gfp=0x%08lx grid_filter_gain=0x%08lx grid_contrast_gain=0x%08lx\r\n",
    (unsigned long)DDR_READ_GAIN_BASEADDR,
    (unsigned long)DDR_READ_PTNT_0_BASEADDR,
    (unsigned long)SCATTER_BASEADDR,
    (unsigned long)PASSTHROUGH_BASEADDR,
    (unsigned long)PASSTHROUGH_GFP_BASEADDR,
    (unsigned long)MEDIAN_FILTER_BASEADDR,
    (unsigned long)CORRECT_BASEADDR,
    (unsigned long)DDR_WRITE_BASEADDR,
    (unsigned long)DDR_WRITE_GFP_BASEADDR,
    (unsigned long)GRID_FILTER_BASEADDR,
    (unsigned long)GRID_CONTRAST_BASEADDR);
           LOG_L2("BYPASS CFG: gfg=%lu gfp=%lu gcg=%lu gcp=%lu sc=%lu mf=%lu cor=%lu\r\n",
               (unsigned long)BYPASS_GFG,
               (unsigned long)BYPASS_GFP,
               (unsigned long)BYPASS_GCG,
               (unsigned long)BYPASS_GCP,
               (unsigned long)BYPASS_SC,
               (unsigned long)BYPASS_MF,
               (unsigned long)BYPASS_COR);
        LOG_L2("DATA MAP: PL_DDR_GAIN=0x%08lx%08lx PL_DDR_PTNT=0x%08lx%08lx PL_DDR_SINK=0x%08lx%08lx\r\n",
            (unsigned long)(((u64)PL_DDR_GAIN_BASE) >> 32), (unsigned long)(((u64)PL_DDR_GAIN_BASE) & 0xFFFFFFFFULL),
            (unsigned long)(((u64)PL_DDR_PTNT_BASE) >> 32), (unsigned long)(((u64)PL_DDR_PTNT_BASE) & 0xFFFFFFFFULL),
            (unsigned long)(((u64)PL_DDR_SINK_BASE) >> 32), (unsigned long)(((u64)PL_DDR_SINK_BASE) & 0xFFFFFFFFULL));

    //log_kernel_ready_idle_done("after");

    g_kernels_inited = true;
    return XST_SUCCESS;
}

static void dump_process_axilite_regs(conn_ctx_t *ctx, int dump_taps)
    {
        static u32 s_process_dump_seq = 0U;
        u32 seq = ++s_process_dump_seq;
        u32 tap_word;
        u32 tap_idx;
        s16 x_taps[21];
        s16 y_taps[35];

        u64 rd_gain_base  = g_ddr_read_gain.Control_BaseAddress;
        u64 rd_ptnt0_base  = g_ddr_read_ptnt_0.Control_BaseAddress;
        u64 gf_base        = g_grid_filter.Ctrl_BaseAddress;
        u64 gf_ptnt_base   = g_grid_filter_ptnt.Ctrl_BaseAddress;
        u64 gs_base        = g_grid_contrast.Ctrl_BaseAddress;
        u64 gs_ptnt_base   = g_grid_contrast_ptnt.Ctrl_BaseAddress;
        u64 sc_base        = g_scatter.Ctrl_BaseAddress;
        u64 pt_base        = g_passthrough.Ctrl_bus_BaseAddress;
        u64 pt_gfp_base    = g_passthrough_gfp.Ctrl_bus_BaseAddress;
        u64 mf_base        = g_median_filter.Ctrl_BaseAddress;
        u64 cor_base       = g_correct.Ctrl_bus_BaseAddress;
        u64 wg_base        = g_ddr_write.Control_BaseAddress;
        u64 dw_gfp_base    = g_ddr_write_gfp.Control_BaseAddress;

        u32 rd_gain_ctrl = XDdr_read_ReadReg(rd_gain_base, XDDR_READ_CONTROL_ADDR_AP_CTRL);
        u32 rd_gain_h = XDdr_read_ReadReg(rd_gain_base, XDDR_READ_CONTROL_ADDR_IMAGE_HEIGHT_DATA);
        u32 rd_gain_w = XDdr_read_ReadReg(rd_gain_base, XDDR_READ_CONTROL_ADDR_IMAGE_WIDTH_DATA);

        u32 rd_ptnt0_ctrl = XDdr_read_ReadReg(rd_ptnt0_base, XDDR_READ_CONTROL_ADDR_AP_CTRL);
        u32 rd_ptnt0_h = XDdr_read_ReadReg(rd_ptnt0_base, XDDR_READ_CONTROL_ADDR_IMAGE_HEIGHT_DATA);
        u32 rd_ptnt0_w = XDdr_read_ReadReg(rd_ptnt0_base, XDDR_READ_CONTROL_ADDR_IMAGE_WIDTH_DATA);

        u32 gf_ctrl = XGrid_filter_ReadReg(gf_base, XGRID_FILTER_CTRL_ADDR_AP_CTRL);
        u32 gf_bp = grid_filter_get_bypass(&g_grid_filter);
        u32 gf_cy0 = XGrid_filter_ReadReg(gf_base, XGRID_FILTER_CTRL_ADDR_COEFF_Y_PACKED_DATA);
        u32 gf_cx0 = XGrid_filter_ReadReg(gf_base, XGRID_FILTER_CTRL_ADDR_COEFF_X_PACKED_DATA);
        u32 gf_ptnt_ctrl = XGrid_filter_ReadReg(gf_ptnt_base, XGRID_FILTER_CTRL_ADDR_AP_CTRL);
        u32 gf_ptnt_bp = grid_filter_get_bypass(&g_grid_filter_ptnt);
        u32 gs_ctrl = XGrid_contrast_ReadReg(gs_base, XGRID_CONTRAST_CTRL_ADDR_AP_CTRL);
        u32 gs_bp = grid_contrast_get_bypass(&g_grid_contrast);
        u32 gs_g0 = XGrid_contrast_ReadReg(gs_base, XGRID_CONTRAST_CTRL_ADDR_GAUSS_1D_PACKED_DATA);
        u32 gs_g1 = XGrid_contrast_ReadReg(gs_base, XGRID_CONTRAST_CTRL_ADDR_GAUSS_1D_PACKED_DATA + 4);
        u32 gs_h = XGrid_contrast_ReadReg(gs_base, XGRID_CONTRAST_CTRL_ADDR_IMAGE_HEIGHT_DATA);
        u32 gs_w = XGrid_contrast_ReadReg(gs_base, XGRID_CONTRAST_CTRL_ADDR_IMAGE_WIDTH_DATA);
        u32 gs_ptnt_ctrl = XGrid_contrast_ReadReg(gs_ptnt_base, XGRID_CONTRAST_CTRL_ADDR_AP_CTRL);
        u32 gs_ptnt_bp = grid_contrast_get_bypass(&g_grid_contrast_ptnt);

        u32 sc_ctrl = XScatter_ReadReg(sc_base, XSCATTER_CTRL_ADDR_AP_CTRL);
        u32 sc_bp = scatter_get_bypass(&g_scatter);
        u32 sc_h = XScatter_ReadReg(sc_base, XSCATTER_CTRL_ADDR_IMAGE_HEIGHT_DATA);
        u32 sc_w = XScatter_ReadReg(sc_base, XSCATTER_CTRL_ADDR_IMAGE_WIDTH_DATA);

        u32 pt_ctrl = XBlock_passthrough_ReadReg(pt_base, XBLOCK_PASSTHROUGH_CTRL_BUS_ADDR_AP_CTRL);
        u32 pt_h = XBlock_passthrough_ReadReg(pt_base, XBLOCK_PASSTHROUGH_CTRL_BUS_ADDR_IMAGE_HEIGHT_DATA);
        u32 pt_w = XBlock_passthrough_ReadReg(pt_base, XBLOCK_PASSTHROUGH_CTRL_BUS_ADDR_IMAGE_WIDTH_DATA);

        u32 pt_gfp_ctrl = XBlock_passthrough_ReadReg(pt_gfp_base, XBLOCK_PASSTHROUGH_CTRL_BUS_ADDR_AP_CTRL);
        u32 pt_gfp_h = XBlock_passthrough_ReadReg(pt_gfp_base, XBLOCK_PASSTHROUGH_CTRL_BUS_ADDR_IMAGE_HEIGHT_DATA);
        u32 pt_gfp_w = XBlock_passthrough_ReadReg(pt_gfp_base, XBLOCK_PASSTHROUGH_CTRL_BUS_ADDR_IMAGE_WIDTH_DATA);

        u32 mf_ctrl = XMedian_filter_ReadReg(mf_base, XMEDIAN_FILTER_CTRL_ADDR_AP_CTRL);
        u32 mf_bp = median_filter_get_bypass(&g_median_filter);
        u32 mf_h = XMedian_filter_ReadReg(mf_base, XMEDIAN_FILTER_CTRL_ADDR_IMAGE_HEIGHT_DATA);
        u32 mf_w = XMedian_filter_ReadReg(mf_base, XMEDIAN_FILTER_CTRL_ADDR_IMAGE_WIDTH_DATA);

        u32 cor_ctrl = XCorrect_ReadReg(cor_base, XCORRECT_CTRL_BUS_ADDR_AP_CTRL);
        u32 cor_bp = correct_get_bypass(&g_correct);
        u32 cor_h = XCorrect_ReadReg(cor_base, XCORRECT_CTRL_BUS_ADDR_IMAGE_HEIGHT_DATA);
        u32 cor_w = XCorrect_ReadReg(cor_base, XCORRECT_CTRL_BUS_ADDR_IMAGE_WIDTH_DATA);

        u32 wg_ctrl = XDdr_write_ReadReg(wg_base, XDDR_WRITE_CONTROL_ADDR_AP_CTRL);
        u32 wg_h = XDdr_write_ReadReg(wg_base, XDDR_WRITE_CONTROL_ADDR_IMAGE_HEIGHT_DATA);
        u32 wg_w = XDdr_write_ReadReg(wg_base, XDDR_WRITE_CONTROL_ADDR_IMAGE_WIDTH_DATA);

        u32 dw_gfp_ctrl = XDdr_write_ReadReg(dw_gfp_base, XDDR_WRITE_CONTROL_ADDR_AP_CTRL);
        u32 dw_gfp_h = XDdr_write_ReadReg(dw_gfp_base, XDDR_WRITE_CONTROL_ADDR_IMAGE_HEIGHT_DATA);
        u32 dw_gfp_w = XDdr_write_ReadReg(dw_gfp_base, XDDR_WRITE_CONTROL_ADDR_IMAGE_WIDTH_DATA);

        LOG_L2("AXIL[%lu] seen: ddr_read_gain=%d ddr_read_ptnt_0=%d grid_filter_gain=%d grid_filter_ptnt=%d grid_contrast_gain=%d grid_contrast_ptnt=%d scatter_0=%d passthrough_0=%d median_filter_0=%d correct_0=%d ddr_write=%d passthrough_gfp=%d ddr_write_gfp=%d\r\n",
            (unsigned long)seq,
            (int)ctx->ddr_read_gain_done_seen,
            (int)ctx->ddr_read_ptnt_0_done_seen,
            (int)ctx->grid_filter_gain_done_seen,
            (int)ctx->grid_filter_ptnt_done_seen,
            (int)ctx->grid_contrast_gain_done_seen,
            (int)ctx->grid_contrast_ptnt_done_seen,
            (int)ctx->scatter_done_seen,
            (int)ctx->passthrough_done_seen,
            (int)ctx->median_filter_done_seen,
            (int)ctx->correct_done_seen,
            (int)ctx->ddr_write_done_seen,
            (int)ctx->passthrough_gfp_done_seen,
            (int)ctx->ddr_write_gfp_done_seen);
        LOG_L2("  ddr_read_gain      @0x%08lx%08lx ctrl=0x%08lx h=0x%08lx w=0x%08lx\r\n",
            (unsigned long)(rd_gain_base >> 32), (unsigned long)(rd_gain_base & 0xFFFFFFFFULL),
            (unsigned long)rd_gain_ctrl, (unsigned long)rd_gain_h, (unsigned long)rd_gain_w);
        LOG_L2("  ddr_read_ptnt_0    @0x%08lx%08lx ctrl=0x%08lx h=0x%08lx w=0x%08lx\r\n",
            (unsigned long)(rd_ptnt0_base >> 32), (unsigned long)(rd_ptnt0_base & 0xFFFFFFFFULL),
            (unsigned long)rd_ptnt0_ctrl, (unsigned long)rd_ptnt0_h, (unsigned long)rd_ptnt0_w);
        LOG_L2("  grid_filter_gain   @0x%08lx%08lx ctrl=0x%08lx bp=0x%08lx cy0=0x%08lx cx0=0x%08lx\r\n",
            (unsigned long)(gf_base >> 32), (unsigned long)(gf_base & 0xFFFFFFFFULL),
            (unsigned long)gf_ctrl, (unsigned long)gf_bp,
            (unsigned long)gf_cy0, (unsigned long)gf_cx0);
        LOG_L2("  grid_filter_ptnt   @0x%08lx%08lx ctrl=0x%08lx bp=0x%08lx\r\n",
            (unsigned long)(gf_ptnt_base >> 32), (unsigned long)(gf_ptnt_base & 0xFFFFFFFFULL),
            (unsigned long)gf_ptnt_ctrl, (unsigned long)gf_ptnt_bp);
        LOG_L2("  grid_contrast_gain @0x%08lx%08lx ctrl=0x%08lx bp=0x%08lx g0=0x%08lx g1=0x%08lx h=0x%08lx w=0x%08lx\r\n",
            (unsigned long)(gs_base >> 32), (unsigned long)(gs_base & 0xFFFFFFFFULL),
            (unsigned long)gs_ctrl, (unsigned long)gs_bp, (unsigned long)gs_g0, (unsigned long)gs_g1, (unsigned long)gs_h, (unsigned long)gs_w);
        LOG_L2("  grid_contrast_ptnt @0x%08lx%08lx ctrl=0x%08lx bp=0x%08lx\r\n",
            (unsigned long)(gs_ptnt_base >> 32), (unsigned long)(gs_ptnt_base & 0xFFFFFFFFULL),
            (unsigned long)gs_ptnt_ctrl, (unsigned long)gs_ptnt_bp);
        LOG_L2("  scatter_0          @0x%08lx%08lx ctrl=0x%08lx bp=0x%08lx h=0x%08lx w=0x%08lx\r\n",
            (unsigned long)(sc_base >> 32), (unsigned long)(sc_base & 0xFFFFFFFFULL),
            (unsigned long)sc_ctrl, (unsigned long)sc_bp, (unsigned long)sc_h, (unsigned long)sc_w);
        LOG_L2("  passthrough_0      @0x%08lx%08lx ctrl=0x%08lx h=0x%08lx w=0x%08lx\r\n",
            (unsigned long)(pt_base >> 32), (unsigned long)(pt_base & 0xFFFFFFFFULL),
            (unsigned long)pt_ctrl, (unsigned long)pt_h, (unsigned long)pt_w);
        LOG_L2("  passthrough_gfp    @0x%08lx%08lx ctrl=0x%08lx h=0x%08lx w=0x%08lx\r\n",
            (unsigned long)(pt_gfp_base >> 32), (unsigned long)(pt_gfp_base & 0xFFFFFFFFULL),
            (unsigned long)pt_gfp_ctrl, (unsigned long)pt_gfp_h, (unsigned long)pt_gfp_w);
        LOG_L2("  median_filter_0    @0x%08lx%08lx ctrl=0x%08lx bp=0x%08lx h=0x%08lx w=0x%08lx\r\n",
            (unsigned long)(mf_base >> 32), (unsigned long)(mf_base & 0xFFFFFFFFULL),
            (unsigned long)mf_ctrl, (unsigned long)mf_bp, (unsigned long)mf_h, (unsigned long)mf_w);
        LOG_L2("  correct_0         @0x%08lx%08lx ctrl=0x%08lx bp=0x%08lx h=0x%08lx w=0x%08lx\r\n",
            (unsigned long)(cor_base >> 32), (unsigned long)(cor_base & 0xFFFFFFFFULL),
            (unsigned long)cor_ctrl, (unsigned long)cor_bp, (unsigned long)cor_h, (unsigned long)cor_w);
        LOG_L2("  ddr_write         @0x%08lx%08lx ctrl=0x%08lx h=0x%08lx w=0x%08lx\r\n",
            (unsigned long)(wg_base >> 32), (unsigned long)(wg_base & 0xFFFFFFFFULL),
            (unsigned long)wg_ctrl, (unsigned long)wg_h, (unsigned long)wg_w);
        LOG_L2("  ddr_write_gfp     @0x%08lx%08lx ctrl=0x%08lx h=0x%08lx w=0x%08lx\r\n",
            (unsigned long)(dw_gfp_base >> 32), (unsigned long)(dw_gfp_base & 0xFFFFFFFFULL),
            (unsigned long)dw_gfp_ctrl, (unsigned long)dw_gfp_h, (unsigned long)dw_gfp_w);

        if (dump_taps) {
            for (tap_idx = 0U; tap_idx < 21U; tap_idx += 2U) {
                tap_word = XGrid_filter_ReadReg(gf_base, XGRID_FILTER_CTRL_ADDR_COEFF_X_PACKED_DATA + ((tap_idx >> 1) * 4U));
                x_taps[tap_idx] = (s16)(tap_word & 0xFFFFU);
                if ((tap_idx + 1U) < 21U) {
                    x_taps[tap_idx + 1U] = (s16)((tap_word >> 16) & 0xFFFFU);
                }
            }

            for (tap_idx = 0U; tap_idx < 35U; tap_idx += 2U) {
                tap_word = XGrid_filter_ReadReg(gf_base, XGRID_FILTER_CTRL_ADDR_COEFF_Y_PACKED_DATA + ((tap_idx >> 1) * 4U));
                y_taps[tap_idx] = (s16)(tap_word & 0xFFFFU);
                if ((tap_idx + 1U) < 35U) {
                    y_taps[tap_idx + 1U] = (s16)((tap_word >> 16) & 0xFFFFU);
                }
            }

            LOG_L2("  grid taps X[21]:");
            for (tap_idx = 0U; tap_idx < 21U; ++tap_idx) {
                LOG_L2(" %d", (int)x_taps[tap_idx]);
            }
            LOG_L2("\r\n");

            LOG_L2("  grid taps Y[35]:");
            for (tap_idx = 0U; tap_idx < 35U; ++tap_idx) {
                LOG_L2(" %d", (int)y_taps[tap_idx]);
            }
            LOG_L2("\r\n");

            /* Dump contrast 1D taps (packed u16 pairs) */
            {
                s16 c_taps[25];
                u32 c_word;
                for (tap_idx = 0U; tap_idx < 25U; tap_idx += 2U) {
                    c_word = XGrid_contrast_ReadReg(gs_base, XGRID_CONTRAST_CTRL_ADDR_GAUSS_1D_PACKED_DATA + ((tap_idx >> 1) * 4U));
                    c_taps[tap_idx] = (s16)(c_word & 0xFFFFU);
                    if ((tap_idx + 1U) < 25U) {
                        c_taps[tap_idx + 1U] = (s16)((c_word >> 16) & 0xFFFFU);
                    }
                }

                LOG_L2("  grid_contrast taps[25]:");
                for (tap_idx = 0U; tap_idx < 25U; ++tap_idx) {
                    LOG_L2(" %d", (int)c_taps[tap_idx]);
                }
                LOG_L2("\r\n");
            }
        }
    }

static void log_kernel_ap_ctrl_one_line(void)
{
    u32 rdg  = XDdr_read_ReadReg(g_ddr_read_gain.Control_BaseAddress,  XDDR_READ_CONTROL_ADDR_AP_CTRL);
    u32 rdp0 = XDdr_read_ReadReg(g_ddr_read_ptnt_0.Control_BaseAddress, XDDR_READ_CONTROL_ADDR_AP_CTRL);
    u32 gfg  = XGrid_filter_ReadReg(g_grid_filter.Ctrl_BaseAddress,      XGRID_FILTER_CTRL_ADDR_AP_CTRL);
    u32 gfp  = XGrid_filter_ReadReg(g_grid_filter_ptnt.Ctrl_BaseAddress, XGRID_FILTER_CTRL_ADDR_AP_CTRL);
    u32 gcg  = XGrid_contrast_ReadReg(g_grid_contrast.Ctrl_BaseAddress,      XGRID_CONTRAST_CTRL_ADDR_AP_CTRL);
    u32 gcp  = XGrid_contrast_ReadReg(g_grid_contrast_ptnt.Ctrl_BaseAddress, XGRID_CONTRAST_CTRL_ADDR_AP_CTRL);
    u32 sc   = XScatter_ReadReg(g_scatter.Ctrl_BaseAddress,         XSCATTER_CTRL_ADDR_AP_CTRL);
    u32 pt   = XBlock_passthrough_ReadReg(g_passthrough.Ctrl_bus_BaseAddress, XBLOCK_PASSTHROUGH_CTRL_BUS_ADDR_AP_CTRL);
    u32 mf   = XMedian_filter_ReadReg(g_median_filter.Ctrl_BaseAddress, XMEDIAN_FILTER_CTRL_ADDR_AP_CTRL);
    u32 cor  = XCorrect_ReadReg(g_correct.Ctrl_bus_BaseAddress,       XCORRECT_CTRL_BUS_ADDR_AP_CTRL);
    u32 dw      = XDdr_write_ReadReg(g_ddr_write.Control_BaseAddress,       XDDR_WRITE_CONTROL_ADDR_AP_CTRL);
    u32 pt_gfp_ = XBlock_passthrough_ReadReg(g_passthrough_gfp.Ctrl_bus_BaseAddress, XBLOCK_PASSTHROUGH_CTRL_BUS_ADDR_AP_CTRL);
    u32 dw_gfp_ = XDdr_write_ReadReg(g_ddr_write_gfp.Control_BaseAddress,    XDDR_WRITE_CONTROL_ADDR_AP_CTRL);

    LOG_L2("AP_CTRL: rdg=0x%08lx rdp0=0x%08lx gfg=0x%08lx gfp=0x%08lx gcg=0x%08lx gcp=0x%08lx sc=0x%08lx pt=0x%08lx mf=0x%08lx cor=0x%08lx dw=0x%08lx pt_gfp=0x%08lx dw_gfp=0x%08lx\r\n",
           (unsigned long)rdg,    (unsigned long)rdp0,
           (unsigned long)gfg,    (unsigned long)gfp,
           (unsigned long)gcg,    (unsigned long)gcp,
           (unsigned long)sc,     (unsigned long)pt,
           (unsigned long)mf,     (unsigned long)cor,
           (unsigned long)dw,     (unsigned long)pt_gfp_,
           (unsigned long)dw_gfp_);
}

void print_app_header()
{
#if (LWIP_IPV6==0)
	LOG_L2("\n\r\n\r-----lwIP TCP server ------\n\r");
#else
    LOG_L2("\n\r\n\r-----lwIPv6 TCP server ------\n\r");
#endif
    LOG_L2("TCP packets sent to port 6001 will be echoed back via DMA\n\r");
}

/* lwIP helpers */

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
            
            case WAIT_HEADER: {
                DEBUG_PRINT("State: WAIT_HEADER\r\n");

                /* Read 4-byte header length */
                if (ctx->expected_header_len == 0) {
                    if (len >= 4) {
                        uint32_t be_len;
                        memcpy(&be_len, data, 4);
                        ctx->expected_header_len = ntohl(be_len);
                        data += 4; len -= 4;
                        // tcp_recved(tpcb, 4);                    // ACK the 4 bytes

                        DEBUG_PRINT("Header length: %u\r\n", ctx->expected_header_len);

                    } else {
                        DEBUG_PRINT("Partial header length received!\r\n");
						len = 0;
                        break;
                    }
                }

                if (ctx->expected_header_len > 0) {
                    if (ctx->expected_header_len > (uint32_t)sizeof(ctx->header_buf)) {
                        LOG_ERR("ERROR: header too large: %u > %u\r\n",
                                ctx->expected_header_len, (unsigned)sizeof(ctx->header_buf));
                        return ERR_OK;
                    }

                    uint32_t copy_len = MIN(len, ctx->expected_header_len - ctx->header_received);
                    memcpy(ctx->header_buf + ctx->header_received, data, copy_len);
                    ctx->header_received += copy_len;
                    data += copy_len; len -= copy_len;
                    // tcp_recved(tpcb, (u16_t)copy_len);          // ACK header bytes

                    if (ctx->header_received == ctx->expected_header_len) {
                        /* Stage buffers in PS DDR. */
                        ctx->pixel_buf_src = pixel_buf_src;
                        ctx->pixel_buf_dst = pixel_buf_dst;

                        /* New transaction: clear processing/sending state. */
                        ctx->dma_done = false;
                        ctx->proc_phase = 0;
                        ctx->dma_started = false;
                        ctx->dma_phase = 0;
                        ctx->tx_offset = 0;
                        ctx->ddr_read_gain_done_seen = false;
                        ctx->ddr_read_ptnt_0_done_seen = false;
                        ctx->ddr_write_done_seen = false;
                        ctx->grid_filter_gain_done_seen = false;
                        ctx->grid_filter_ptnt_done_seen = false;
                        ctx->grid_contrast_gain_done_seen = false;
                        ctx->grid_contrast_ptnt_done_seen = false;
                        ctx->scatter_done_seen = false;
                        ctx->passthrough_done_seen = false;
                        ctx->passthrough_gfp_done_seen = false;
                        ctx->median_filter_done_seen = false;
                        ctx->correct_done_seen = false;
                        ctx->ddr_write_gfp_done_seen = false;

                        ctx->expected_input_bytes = 0;
                        ctx->expected_input_proc_bytes = 0;
                        ctx->expected_output_bytes = 0;
                        ctx->input_bytes_received = 0;
                        ctx->pixel_chunk_remaining = 0;
                        ctx->pixel_chunk_len = 0;

                        ctx->fxd_endian = 0;
                        ctx->fxd_type_code_in = 0;
                        ctx->fxd_frames = 0;
                        ctx->fxd_commentlen = 0;

                        /* FXD/FDXD only. */
                        if (ctx->expected_header_len < FXD_HEADER_FIXED_BYTES) {
                            LOG_ERR("ERROR: FXD header too small: %u < %u\r\n",
                                    (unsigned)ctx->expected_header_len,
                                    (unsigned)FXD_HEADER_FIXED_BYTES);
                            return ERR_OK;
                        }

                        char endian = 0;
                        if (!fxd_detect_endian(ctx->header_buf, ctx->expected_header_len, &endian)) {
                            LOG_ERR("ERROR: not an FXD/FDXD header (bad magic)\r\n");
                            return ERR_OK;
                        }
                        if (endian != '<') {
                            LOG_ERR("ERROR: FXD big-endian not supported in this build\r\n");
                            return ERR_OK;
                        }

                        u32 magic = fxd_rd_i32(ctx->header_buf + 0, endian);
                        u32 cols = fxd_rd_i32(ctx->header_buf + 4, endian);
                        u32 rows = fxd_rd_i32(ctx->header_buf + 8, endian);
                        u32 frames = fxd_rd_i32(ctx->header_buf + 12, endian);
                        u32 type_code = fxd_rd_i32(ctx->header_buf + 16, endian);
                        u32 commentlen = fxd_rd_i32(ctx->header_buf + 32, endian);

                        if (ctx->expected_header_len < (FXD_HEADER_FIXED_BYTES + commentlen)) {
                            LOG_ERR("ERROR: FXD header incomplete: hdr_len=%u, needs %u (128+comment)\r\n",
                                    (unsigned)ctx->expected_header_len,
                                    (unsigned)(FXD_HEADER_FIXED_BYTES + commentlen));
                            return ERR_OK;
                        }
                        if (cols == 0U || rows == 0U || frames == 0U) {
                            LOG_ERR("ERROR: FXD invalid geometry: cols=%u rows=%u frames=%u\r\n",
                                    (unsigned)cols, (unsigned)rows, (unsigned)frames);
                            return ERR_OK;
                        }
 
                        if (type_code != FXD_TYPE_INT16) {
                            LOG_ERR("ERROR: FXD input type_code=%u; expected int16 (1)\r\n",
                                    (unsigned)type_code);
                            return ERR_OK;
                        }

                        u64 npix = (u64)cols * (u64)rows * (u64)frames;
                        u64 in_bytes = npix * 2ULL;
                        u64 proc_rows = (u64)IMAGE_HEIGHT;
                        u64 proc_npixels = (u64)cols * proc_rows * (u64)frames;
                        u64 proc_in_bytes = proc_npixels * 2ULL;
                        u64 out_bytes = proc_npixels * 2ULL;

                        /* ---- Validate against HLS kernel assumptions ----
                         * ddr_read/ddr_write burst in fixed groups and rely on row alignment.
                         * If these constraints are violated, corruption is expected.
                         */
                        u64 h_total = (u64)rows * (u64)frames; /* frames are stacked as rows in SW */

                        if (cols > 0xFFFU || h_total > 0xFFFULL) {
                            LOG_ERR("ERROR: geometry exceeds 12-bit regs: cols=%u height(rows*frames)=%llu\r\n",
                                    (unsigned)cols, (unsigned long long)h_total);
                            return ERR_OK;
                        }
                        if (cols > (u32)HLS_DDR_MAX_WIDTH || h_total > (u64)HLS_DDR_MAX_HEIGHT) {
                            LOG_ERR("ERROR: geometry exceeds HLS MEM_DEPTH: cols=%u height=%llu (max %u x %u)\r\n",
                                    (unsigned)cols,
                                    (unsigned long long)h_total,
                                    (unsigned)HLS_DDR_MAX_WIDTH,
                                    (unsigned)HLS_DDR_MAX_HEIGHT);
                            return ERR_OK;
                        }


                        if (out_bytes > (u64)MAX_IMAGE_SIZE) {
                            LOG_ERR("ERROR: FXD int16 image too large: %llu > %u\r\n",
                                    (unsigned long long)out_bytes,
                                    (unsigned)MAX_IMAGE_SIZE);
                            return ERR_OK;
                        }

                        ctx->fxd_endian = endian;
                        ctx->fxd_type_code_in = type_code;
                        ctx->fxd_frames = frames;
                        ctx->fxd_commentlen = commentlen;
                        ctx->img_width = cols;
                        ctx->img_height = (u32)proc_rows;
                        ctx->expected_input_bytes = (u32)in_bytes;
                        ctx->expected_input_proc_bytes = (u32)proc_in_bytes;
                        ctx->expected_output_bytes = (u32)out_bytes;

                        /* Set reply header magic/type/rows for processed output. */
                        if (magic == FXD_MAGIC_OLD) {
                            fxd_set_magic(ctx->header_buf, endian, FXD_MAGIC_NEW);
                        }
                        fxd_set_type_code(ctx->header_buf, endian, FXD_TYPE_INT16);
                        fxd_set_rows(ctx->header_buf, endian, (u32)proc_rows);

                           LOG_L3("Received FXD header: %lux%lu -> process %lux%lu frames=%lu in=%luB(proc=%luB) out=%luB\r\n",
                               (unsigned long)cols,
                               (unsigned long)rows,
                               (unsigned long)cols,
                               (unsigned long)proc_rows,
                               (unsigned long)frames,
                               (unsigned long)ctx->expected_input_bytes,
                               (unsigned long)ctx->expected_input_proc_bytes,
                               (unsigned long)ctx->expected_output_bytes);

                        /* Send ACK and move on */
                        tcp_write(tpcb, "ACK", 3, TCP_WRITE_FLAG_COPY);
                        tcp_output(tpcb);
                        DEBUG_PRINT("ACK sent to PC\r\n");
                        ctx->state = WAIT_PIXELS;
                    }
                }
                break;
            }
            
            case WAIT_PIXELS: {
                if (ctx->input_bytes_received == 0 && ctx->pixel_chunk_remaining == 0) {
                    LOG_L2("State: WAIT_PIXELS (expecting %u bytes)\r\n", ctx->expected_input_bytes);
                } else {
                    LOG_L1("State: WAIT_PIXELS\r\n");
                }

                /* If current chunk length is not set, read it */
                if (ctx->pixel_chunk_remaining == 0) {
                    if (ctx->input_bytes_received >= ctx->expected_input_bytes) {

                        /* New transaction: clear processing state. */
                        ctx->dma_done = false;
                        ctx->proc_phase = 0;
                        ctx->dma_started = false;
                        ctx->dma_phase = 0;
                        ctx->tx_offset = 0;

                        ctx->state = SEND_TO_PL;
                        LOG_L3("Received image payload (%u bytes); starting pipeline\r\n", ctx->expected_input_bytes);
                        #if MONITOR_LWIP_RX
                        {
                            uint64_t elapsed_ticks = timer_get_count() - ctx->rx_t_start;
                            uint64_t elapsed_us = elapsed_ticks * 1000000ULL / timer_get_freq_hz();
                            LOG_L3("LWIP RX: %u bytes in %llu us\r\n",
                                   (unsigned)ctx->expected_input_bytes,
                                   (unsigned long long)elapsed_us);
                        }
                        #endif
                        break;
                    }
                    if (len < 4) { 
                        len = 0;
                        break; }           // need more bytes for the length

                    uint32_t be_len;
                    memcpy(&be_len, data, 4);
                    ctx->pixel_chunk_len = ntohl(be_len);
                    data += 4; len -= 4;
                    // tcp_recved(tpcb, 4);               // ACK the chunk length

                if (ctx->pixel_chunk_len > g_rx_chunk_max_bytes) {
                    LOG_ERR("ERROR: RX chunk too large: %u (max %u)\r\n",
                        ctx->pixel_chunk_len, g_rx_chunk_max_bytes);
                    /* Protocol violation / memory protection: stop processing this pbuf. */
                    len = 0;
                    break;
                }

                    /* Bound the remaining by expected total */
                    ctx->pixel_chunk_remaining = MIN(ctx->pixel_chunk_len,
                                                    ctx->expected_input_bytes - ctx->input_bytes_received);

                    DEBUG_PRINT("New pixel chunk: %u bytes\r\n", ctx->pixel_chunk_remaining);
                }

                /* Copy available payload for the current chunk */
                uint32_t to_copy = MIN(len, ctx->pixel_chunk_remaining);
                if (to_copy == 0) { break; }

                /* Always store the incoming payload as-is in src buffer.
                 * For FXD this is raw int16 bytes; the pipeline will expand to float32.
                 */
                memcpy(ctx->pixel_buf_src + ctx->input_bytes_received, data, to_copy);

                #if MONITOR_LWIP_RX
                if (ctx->input_bytes_received == 0) {
                    ctx->rx_t_start = timer_get_count();
                }
                #endif

                ctx->input_bytes_received += to_copy;
                ctx->pixel_chunk_remaining -= to_copy;
                data += to_copy; len -= to_copy;

                // tcp_recved(tpcb, (u16_t)to_copy);      // ACK payload bytes

                if (ctx->pixel_chunk_remaining == 0) {
                        DEBUG_PRINT("Chunk complete. Total pixels: %u / %u\r\n",
                            ctx->input_bytes_received, ctx->expected_input_bytes);
                        DEBUG_PRINT("len=%d\r\n", len);
                }


                if (ctx->input_bytes_received >= ctx->expected_input_bytes) {

                    /* FXD input is int16, must be an even byte count. */
                    if ((ctx->expected_input_bytes & 1U) != 0U) {
                        LOG_ERR("ERROR: FXD expected_input_bytes is odd (%u)\r\n",
                                (unsigned)ctx->expected_input_bytes);
                        return ERR_OK;
                    }
                    
                    /* New transaction: clear processing state. */
                    ctx->dma_done = false;
                    ctx->proc_phase = 0;
                    ctx->dma_started = false;
                    ctx->dma_phase = 0;
                    ctx->tx_offset = 0;
                    ctx->ddr_read_gain_done_seen = false;
                    ctx->ddr_read_ptnt_0_done_seen = false;
                    ctx->ddr_write_done_seen = false;
                    ctx->grid_filter_gain_done_seen = false;
                    ctx->grid_filter_ptnt_done_seen = false;
                    ctx->grid_contrast_gain_done_seen = false;
                    ctx->grid_contrast_ptnt_done_seen = false;
                    ctx->scatter_done_seen = false;
                    ctx->passthrough_done_seen = false;
                    ctx->passthrough_gfp_done_seen = false;
                    ctx->median_filter_done_seen = false;
                    ctx->correct_done_seen = false;
                    ctx->ddr_write_gfp_done_seen = false;

                    ctx->state = SEND_TO_PL;
                    LOG_L3("Received image payload (%u bytes); starting pipeline\r\n", ctx->expected_input_bytes);
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

			case WAIT_ACK:
            {
                DEBUG_PRINT("State: WAIT_ACK\r\n");
                // Expecting ACK from PC before sending pixel data
                if (len >= 3 && strncmp((const char *)data, "ACK", 3) == 0) {
                    data += 3; len -= 3;
                    DEBUG_PRINT("ACK received from PC, sending pixel data\r\n");
                    // tcp_recved(tpcb, 3);
                    // Seed first pixel chunk to kick off tcp_sent pacing
                    ctx->state = SEND_PIXELS;
                        if (ctx->tx_offset < ctx->expected_output_bytes) {
                            uint32_t remaining = ctx->expected_output_bytes - ctx->tx_offset;
                        u32 sndbuf_payload = (u32)tcp_sndbuf(tpcb);
                        if (sndbuf_payload > 4U) {
                            sndbuf_payload -= 4U;
                        } else {
                            sndbuf_payload = 0U;
                        }
                        uint32_t len_chunk = (uint32_t)MIN(remaining, MIN((uint32_t)g_tx_chunk_bytes, sndbuf_payload));
                        if (len_chunk == 0U) {
                            len = 0;
                            break;
                        }
                        uint32_t len_chunk_net = htonl(len_chunk);
                        if (tcp_write(tpcb, &len_chunk_net, 4, TCP_WRITE_FLAG_COPY) == ERR_OK &&
                            tcp_write(tpcb, ctx->pixel_buf_dst + ctx->tx_offset, len_chunk, TCP_WRITE_FLAG_COPY) == ERR_OK) {
                            tcp_output(tpcb);
                            ctx->tx_offset += len_chunk;
                        }
                    }
                } else {
                    // Not enough data for ACK, or incorrect packet, wait for more
                    len = 0;
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
        if (ctx->tx_offset < ctx->expected_output_bytes) {
            uint32_t remaining = ctx->expected_output_bytes - ctx->tx_offset;
			u32 sndbuf_payload = (u32)tcp_sndbuf(tpcb);
			if (sndbuf_payload > 4U) {
				sndbuf_payload -= 4U;
			} else {
				sndbuf_payload = 0U;
			}
			uint32_t len_chunk = (uint32_t)MIN(remaining, MIN((uint32_t)g_tx_chunk_bytes, sndbuf_payload));
			if (len_chunk == 0U) {
				return ERR_OK;
			}
            uint32_t len_chunk_net = htonl(len_chunk);
            if (tcp_write(tpcb, &len_chunk_net, 4, TCP_WRITE_FLAG_COPY) == ERR_OK &&
                tcp_write(tpcb, ctx->pixel_buf_dst + ctx->tx_offset, len_chunk, TCP_WRITE_FLAG_COPY) == ERR_OK) {
                tcp_output(tpcb);
                ctx->tx_offset += len_chunk;
            }
        }
        if (ctx->tx_offset >= ctx->expected_output_bytes) {
            LOG_L3("Image sent back (%u bytes)\r\n", ctx->expected_output_bytes);
            ctx->state = WAIT_HEADER;
            ctx->expected_header_len = 0;
            ctx->header_received = 0;
            ctx->expected_input_bytes = 0;
            ctx->expected_input_proc_bytes = 0;
            ctx->expected_output_bytes = 0;
            ctx->input_bytes_received = 0;
            ctx->tx_offset = 0;
            ctx->dma_started = false;
            ctx->fxd_endian = 0;
            ctx->fxd_type_code_in = 0;
            ctx->fxd_frames = 0;
            ctx->fxd_commentlen = 0;
            ctx->got_gain = false;
        }
    }
    return ERR_OK;
}

static err_t poll_callback(void *arg, struct tcp_pcb *tpcb)
{
    conn_ctx_t *ctx = (conn_ctx_t *)arg;

    switch (ctx->state) {
    case SEND_TO_PL: {
        DEBUG_PRINT("Poll: SEND_TO_PL\r\n");

        /* Validate required buffers and transfer sizes before DMA start. */
        if (!ctx->pixel_buf_src || !ctx->pixel_buf_dst ||
            ctx->expected_input_proc_bytes == 0 || ctx->expected_output_bytes == 0) {
            DEBUG_PRINT("SEND_TO_PL: invalid buffers/size\r\n");
            return ERR_OK;
        }

        /* Kernel init deferred until both images are in PL. */

        /* Per-image transaction resets (safe to re-assert). */
        ctx->dma_done = false;
        ctx->kernels_started = false;

        if (!ctx->dma_started) {
            ctx->dma_started = true;

        /* Flush only the input payload size (FXD: int16 bytes). */
        Xil_DCacheFlushRange((UINTPTR)ctx->pixel_buf_src, ctx->expected_input_proc_bytes);

        UINTPTR dst_base = ctx->got_gain ? (UINTPTR)PL_DDR_PTNT_BASE : (UINTPTR)PL_DDR_GAIN_BASE;

        int s2mm = XAxiDma_SimpleTransfer(&AxiDma,
                            dst_base,
                            ctx->expected_input_proc_bytes,
                            XAXIDMA_DEVICE_TO_DMA);
        int mm2s = XAxiDma_SimpleTransfer(&AxiDma,
                            (UINTPTR)ctx->pixel_buf_src,
                            ctx->expected_input_proc_bytes,
                            XAXIDMA_DMA_TO_DEVICE);

        if (s2mm != XST_SUCCESS || mm2s != XST_SUCCESS) {
            DEBUG_PRINT("SEND_TO_PL: DMA start failed: S2MM=%d MM2S=%d\r\n", s2mm, mm2s);
            ctx->dma_started = false;
            return ERR_OK;
        }

        LOG_L3("COPY PS->PL: bytes=%u dst=0x%016llx\r\n",
                    (unsigned)ctx->expected_input_proc_bytes,
                    (unsigned long long)(u64)dst_base);
#if MONITOR_DMA
        {
            uint64_t t_start = timer_get_count();
            while (XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE) ||
                   XAxiDma_Busy(&AxiDma, XAXIDMA_DEVICE_TO_DMA)) { }
            uint64_t elapsed_ticks = timer_get_count() - t_start;
            uint64_t elapsed_us = elapsed_ticks * 1000000ULL / timer_get_freq_hz();
            LOG_L3("SEND_TO_PL DMA done: bytes=%u time=%llu us\r\n",
                   (unsigned)ctx->expected_input_proc_bytes,
                   (unsigned long long)elapsed_us);
        }
#else
        return ERR_OK;
#endif
    }

    if (XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE) ||
        XAxiDma_Busy(&AxiDma, XAXIDMA_DEVICE_TO_DMA)) {
        return ERR_OK;
    }

    /* DMA finished for this image */
    ctx->dma_started = false;

    if (!ctx->got_gain) {
        /* We just stored the gain image; mark and return to WAIT_HEADER for ptnt */
        ctx->got_gain = true;
        LOG_L3("GAIN image stored to PL DDR (bytes=%u)\r\n", ctx->expected_input_proc_bytes);
        
        /* Reset header state to accept next image */
        ctx->state = WAIT_HEADER;
        ctx->expected_header_len = 0;
        ctx->header_received = 0;
        ctx->expected_input_bytes = 0;
        ctx->expected_input_proc_bytes = 0;
        ctx->expected_output_bytes = 0;
        ctx->input_bytes_received = 0;
        ctx->pixel_chunk_remaining = 0;
        ctx->pixel_chunk_len = 0;

        /* Send ACK and move on */
        
        /* Send ACK to signal PC to send the next image */
        tcp_write(tpcb, "ACK", 3, TCP_WRITE_FLAG_COPY);
        tcp_output(tpcb);
        DEBUG_PRINT("ACK sent to PC\r\n");

        return ERR_OK;
        }

        /* got_gain == true: this is the ptnt image.
         * ACK only after DMA completion so client sees a post-pixels ACK
         * without racing ahead of server state.
         */
        tcp_write(tpcb, "ACK", 3, TCP_WRITE_FLAG_COPY);
        tcp_output(tpcb);
        DEBUG_PRINT("ACK sent to PC\r\n");

        /* Proceed to PROCESS as before */
        LOG_L3("PTNT image stored to PL DDR (bytes=%u)\r\n", ctx->expected_input_proc_bytes);

        ctx->state = PROCESS;
        return ERR_OK;
    }

    case PROCESS: {
    DEBUG_PRINT("Poll: PROCESS\r\n");

    // --------------------------------------------------------------------
    // Start kernels once per transaction
    // --------------------------------------------------------------------
    if (!ctx->kernels_started) {

        // Init kernels once (driver init + coefficient programming, etc.)
        if (!ctx->kernels_ready) {
            if (kernels_init_once() != XST_SUCCESS) {
                DEBUG_PRINT("PROCESS: kernel init failed\r\n");
                return ERR_OK;
            }
            ctx->kernels_ready = true;
        }

        // Program geometry for all blocks
        const u32 proc_w = IMAGE_WIDTH;
        const u32 proc_h = IMAGE_HEIGHT;

        XDdr_read_Set_image_width (&g_ddr_read_gain,   proc_w);
        XDdr_read_Set_image_height(&g_ddr_read_gain,   proc_h);
        XDdr_read_Set_image_width (&g_ddr_read_ptnt_0, proc_w);
        XDdr_read_Set_image_height(&g_ddr_read_ptnt_0, proc_h);

        XGrid_filter_Set_image_width (&g_grid_filter,      proc_w);
        XGrid_filter_Set_image_height(&g_grid_filter,      proc_h);
        XGrid_filter_Set_image_width (&g_grid_filter_ptnt, proc_w);
        XGrid_filter_Set_image_height(&g_grid_filter_ptnt, proc_h);

        XGrid_contrast_Set_image_width (&g_grid_contrast,      proc_w);
        XGrid_contrast_Set_image_height(&g_grid_contrast,      proc_h);
        XGrid_contrast_Set_image_width (&g_grid_contrast_ptnt, proc_w);
        XGrid_contrast_Set_image_height(&g_grid_contrast_ptnt, proc_h);
        
        XScatter_Set_image_width (&g_scatter, proc_w);
        XScatter_Set_image_height(&g_scatter, proc_h);

        XBlock_passthrough_Set_image_width (&g_passthrough, proc_w);
        XBlock_passthrough_Set_image_height(&g_passthrough, proc_h);

        XBlock_passthrough_Set_image_width (&g_passthrough_gfp, proc_w);
        XBlock_passthrough_Set_image_height(&g_passthrough_gfp, proc_h);

        XMedian_filter_Set_image_width (&g_median_filter, proc_w);
        XMedian_filter_Set_image_height(&g_median_filter, proc_h);

        XCorrect_Set_image_width (&g_correct, proc_w);
        XCorrect_Set_image_height(&g_correct, proc_h);

        XDdr_write_Set_image_width (&g_ddr_write, proc_w);
        XDdr_write_Set_image_height(&g_ddr_write, proc_h);

        XDdr_write_Set_image_width (&g_ddr_write_gfp, proc_w);
        XDdr_write_Set_image_height(&g_ddr_write_gfp, proc_h);

    #if DEBUG_KERNELS
        dump_process_axilite_regs(ctx, 1);
        log_kernel_ap_ctrl_one_line();
    #endif

        // ----------------------------------------------------------------
        // Reset SOFTWARE done latches for THIS transaction (critical)
        // ----------------------------------------------------------------
        ctx->ddr_read_gain_done_seen      = false;
        ctx->ddr_read_ptnt_0_done_seen    = false;
        ctx->grid_filter_gain_done_seen   = false;
        ctx->grid_filter_ptnt_done_seen   = false;
        ctx->grid_contrast_gain_done_seen = false;
        ctx->grid_contrast_ptnt_done_seen = false;
        ctx->scatter_done_seen            = false;
        ctx->passthrough_done_seen        = false;
        ctx->passthrough_gfp_done_seen    = false;
        ctx->median_filter_done_seen      = false;
        ctx->correct_done_seen            = false;
        ctx->ddr_write_done_seen          = false;
        ctx->ddr_write_gfp_done_seen      = false;

        // ----------------------------------------------------------------
        // Clear stale HW ap_done (clear-on-read) from any previous run
        // (safe to do once right before starting)
        // ----------------------------------------------------------------
        (void)XDdr_read_IsDone(&g_ddr_read_gain);
        (void)XDdr_read_IsDone(&g_ddr_read_ptnt_0);
        (void)XGrid_filter_IsDone(&g_grid_filter);
        (void)XGrid_filter_IsDone(&g_grid_filter_ptnt);
        (void)XGrid_contrast_IsDone(&g_grid_contrast);
        (void)XGrid_contrast_IsDone(&g_grid_contrast_ptnt);
        (void)XScatter_IsDone(&g_scatter);
        (void)XBlock_passthrough_IsDone(&g_passthrough);
        (void)XBlock_passthrough_IsDone(&g_passthrough_gfp);
        (void)XMedian_filter_IsDone(&g_median_filter);
        (void)XCorrect_IsDone(&g_correct);
        (void)XDdr_write_IsDone(&g_ddr_write);
        (void)XDdr_write_IsDone(&g_ddr_write_gfp);

        // ----------------------------------------------------------------
        // Start kernels (concurrently)
        // ----------------------------------------------------------------
        
        XGrid_filter_Start(&g_grid_filter);
        XDdr_write_Start(&g_ddr_write_gfp);
        XBlock_passthrough_Start(&g_passthrough_gfp);
        XGrid_filter_Start(&g_grid_filter_ptnt);
        XGrid_contrast_Start(&g_grid_contrast);
        XGrid_contrast_Start(&g_grid_contrast_ptnt);
        XScatter_Start(&g_scatter);
        XBlock_passthrough_Start(&g_passthrough);
        XMedian_filter_Start(&g_median_filter);
        XCorrect_Start(&g_correct);
        XDdr_write_Start(&g_ddr_write);
        XDdr_read_Start(&g_ddr_read_gain);
        XDdr_read_Start(&g_ddr_read_ptnt_0);

        LOG_L3("Kernels started (w=%lu h=%lu)\r\n",
             (unsigned long)proc_w,
             (unsigned long)proc_h);

        #if MONITOR_PERFORMANCE
        /* GCC starts here: after all Start() writes and all log prints.
         * This measures only the time from last kernel start to ddr_write done. */
        if (g_axipmon_ready) {
            XAxiPmon_ResetGlobalClkCounter(&g_axipmon);
            u32 _cr = XAxiPmon_ReadReg(g_axipmon.Config.BaseAddress, XAPM_CTL_OFFSET);
            XAxiPmon_WriteReg(g_axipmon.Config.BaseAddress, XAPM_CTL_OFFSET,
                              _cr | XAPM_CR_GCC_ENABLE_MASK);
        }
        #endif

        ctx->kernels_started = true;

        {
            bool wg = false;
            #if MONITOR_PERFORMANCE
            bool wg_gcc_stopped = false;
            #endif

            while (!wg) {
                wg |= (XDdr_write_IsDone(&g_ddr_write) != 0U);
                #if MONITOR_PERFORMANCE
                if (!wg_gcc_stopped && wg) {
                    wg_gcc_stopped = true;
                    if (g_axipmon_ready) {
                        u32 _cr = XAxiPmon_ReadReg(g_axipmon.Config.BaseAddress, XAPM_CTL_OFFSET);
                        XAxiPmon_WriteReg(g_axipmon.Config.BaseAddress, XAPM_CTL_OFFSET,
                                          _cr & ~XAPM_CR_GCC_ENABLE_MASK);
                    }
                }
                #endif
            }

            // Only read upstream ap_done bits when L1 logging is active —
            // they are only needed for the diagnostic Done: line below.
            ctx->ddr_write_done_seen = wg;
            if (g_log_level <= 1) {
                ctx->ddr_read_gain_done_seen      = (XDdr_read_IsDone(&g_ddr_read_gain)            != 0U);
                ctx->ddr_read_ptnt_0_done_seen    = (XDdr_read_IsDone(&g_ddr_read_ptnt_0)          != 0U);
                ctx->grid_filter_gain_done_seen   = (XGrid_filter_IsDone(&g_grid_filter)           != 0U);
                ctx->grid_filter_ptnt_done_seen   = (XGrid_filter_IsDone(&g_grid_filter_ptnt)      != 0U);
                ctx->grid_contrast_gain_done_seen = (XGrid_contrast_IsDone(&g_grid_contrast)       != 0U);
                ctx->grid_contrast_ptnt_done_seen = (XGrid_contrast_IsDone(&g_grid_contrast_ptnt)  != 0U);
                ctx->scatter_done_seen            = (XScatter_IsDone(&g_scatter)                   != 0U);
                ctx->passthrough_done_seen        = (XBlock_passthrough_IsDone(&g_passthrough)     != 0U);
                ctx->median_filter_done_seen      = (XMedian_filter_IsDone(&g_median_filter)       != 0U);
                ctx->correct_done_seen            = (XCorrect_IsDone(&g_correct)                   != 0U);
                ctx->passthrough_gfp_done_seen    = (XBlock_passthrough_IsDone(&g_passthrough_gfp) != 0U);
                ctx->ddr_write_gfp_done_seen      = (XDdr_write_IsDone(&g_ddr_write_gfp)          != 0U);
            }
        }

        #if MONITOR_PERFORMANCE
        if (g_axipmon_ready) {
            /* GCC was already stopped inside the spin-wait when ddr_write first fired.
             * Just read and log the latched count. */
            u32 clk_hi = 0U, clk_lo = 0U;
            XAxiPmon_GetGlobalClkCounter(&g_axipmon, &clk_hi, &clk_lo);
            u64 clk_cycles = ((u64)clk_hi << 32) | (u64)clk_lo;
            LOG_L3("Pipeline PL clock cycles: %llu\r\n", (unsigned long long)clk_cycles);
        }
        #endif

        LOG_L1("Done: rd_gain=%d rd_ptnt_0=%d gf_gain=%d gf_ptnt=%d gs_gain=%d gs_ptnt=%d sc=%d pt=%d mf=%d cor=%d wg=%d pt_gfp=%d dw_gfp=%d\r\n",
               ctx->ddr_read_gain_done_seen,    ctx->ddr_read_ptnt_0_done_seen,
               ctx->grid_filter_gain_done_seen,  ctx->grid_filter_ptnt_done_seen,
               ctx->grid_contrast_gain_done_seen, ctx->grid_contrast_ptnt_done_seen,
               ctx->scatter_done_seen, ctx->passthrough_done_seen, ctx->median_filter_done_seen, ctx->correct_done_seen, ctx->ddr_write_done_seen,
               ctx->passthrough_gfp_done_seen, ctx->ddr_write_gfp_done_seen);

    } // end if (!ctx->kernels_started)

    ctx->state = RECV_FROM_PL;
    ctx->dma_started = false;
    return ERR_OK;
}
    case RECV_FROM_PL: {
        DEBUG_PRINT("Poll: RECV_FROM_PL\r\n");

        if (!ctx->dma_started) {
            ctx->dma_started = true;

            /* Device will write PS DDR dst. */
            Xil_DCacheInvalidateRange((UINTPTR)ctx->pixel_buf_dst, ctx->expected_output_bytes);

            int s2mm = XAxiDma_SimpleTransfer(&AxiDma,
                                (UINTPTR)ctx->pixel_buf_dst,
                                ctx->expected_output_bytes,
                                XAXIDMA_DEVICE_TO_DMA);
            int mm2s = XAxiDma_SimpleTransfer(&AxiDma,
                                (UINTPTR)PL_DDR_SINK_BASE,
                                ctx->expected_output_bytes,
                                XAXIDMA_DMA_TO_DEVICE);

            if (s2mm != XST_SUCCESS || mm2s != XST_SUCCESS) {
                DEBUG_PRINT("RECV_FROM_PL: DMA start failed: S2MM=%d MM2S=%d\r\n", s2mm, mm2s);
                ctx->dma_started = false;
                return ERR_OK;
            }

            LOG_L3("COPY PL->PS: bytes=%u src=0x%016llx\r\n",
                        (unsigned)ctx->expected_output_bytes,
                        (unsigned long long)(u64)PL_DDR_SINK_BASE);
#if MONITOR_DMA
            {
                uint64_t t_start = timer_get_count();
                while (XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE) ||
                       XAxiDma_Busy(&AxiDma, XAXIDMA_DEVICE_TO_DMA)) { }
                uint64_t elapsed_ticks = timer_get_count() - t_start;
                uint64_t elapsed_us = elapsed_ticks * 1000000ULL / timer_get_freq_hz();
                LOG_L3("RECV_FROM_PL DMA done: bytes=%u time=%llu us\r\n",
                       (unsigned)ctx->expected_output_bytes,
                       (unsigned long long)elapsed_us);
            }
#else
            return ERR_OK;
#endif
        }

        if (XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE) ||
            XAxiDma_Busy(&AxiDma, XAXIDMA_DEVICE_TO_DMA)) {
            return ERR_OK;
        }

        ctx->dma_started = false;
        ctx->dma_done = true;

        Xil_DCacheInvalidateRange((UINTPTR)ctx->pixel_buf_dst, ctx->expected_output_bytes);

        ctx->state = SEND_HEADER;
        return ERR_OK;
    }

    case SEND_HEADER: {
        DEBUG_PRINT("Poll: SEND_HEADER\r\n");

        uint32_t hdr_len_net = htonl(ctx->expected_header_len);
        err_t w1 = tcp_write(tpcb, &hdr_len_net, 4, TCP_WRITE_FLAG_COPY);
        err_t w2 = (w1 == ERR_OK)
                 ? tcp_write(tpcb, ctx->header_buf, ctx->expected_header_len, TCP_WRITE_FLAG_COPY)
                 : w1;

        if (w2 == ERR_OK) {
            tcp_output(tpcb);
            LOG_L3("Server header sent\r\n");
            ctx->state = WAIT_ACK;
        } else {
            DEBUG_PRINT("tcp_write(header) returned %d; will retry\r\n", w2);
        }
        return ERR_OK;
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
    ctx->expected_input_proc_bytes = 0;
    ctx->expected_output_bytes = 0;
    ctx->header_received = 0;

    ctx->input_bytes_received = 0;
    ctx->tx_offset = 0;
    ctx->dma_started = false;
    ctx->ack_received = false;
    ctx->ddr_read_gain_done_seen = false;
    ctx->ddr_read_ptnt_0_done_seen = false;
    ctx->ddr_write_done_seen = false;
    ctx->grid_filter_gain_done_seen = false;
    ctx->grid_filter_ptnt_done_seen = false;
    ctx->grid_contrast_gain_done_seen = false;
    ctx->grid_contrast_ptnt_done_seen = false;
    ctx->scatter_done_seen = false;
    ctx->passthrough_done_seen = false;
    ctx->median_filter_done_seen = false;
    ctx->correct_done_seen = false;
    ctx->got_gain = false;
    ctx->pixel_chunk_remaining = 0;
    ctx->pixel_chunk_len = 0;
    memset(ctx->header_buf, 0, sizeof(ctx->header_buf));
    ctx->fxd_endian = 0;
    ctx->fxd_type_code_in = 0;
    ctx->fxd_frames = 0;
    ctx->fxd_commentlen = 0;
    ctx->state = WAIT_HEADER;

    if (abort) tcp_abort(tpcb);
    else tcp_close(tpcb);
}

err_t accept_callback(void *arg, struct tcp_pcb *newpcb, err_t err) {
    conn_ctx_t *ctx = malloc(sizeof(conn_ctx_t));
    memset(ctx, 0, sizeof(conn_ctx_t));

	// Set initial state
    ctx->state = WAIT_HEADER;
    ctx->expected_header_len = 0;
    ctx->expected_input_bytes = 0;
    ctx->expected_input_proc_bytes = 0;
    ctx->expected_output_bytes = 0;
    ctx->header_received = 0;

    ctx->input_bytes_received = 0;
	ctx->tx_offset = 0;
	ctx->ack_received = false;
	ctx->dma_started = false;
    ctx->dma_phase = 0;
    ctx->proc_phase = 0;
    ctx->kernels_ready = false;
    ctx->kernels_started = false;
    ctx->ddr_read_gain_done_seen = false;
    ctx->ddr_read_ptnt_0_done_seen = false;
    ctx->ddr_write_done_seen = false;
    ctx->grid_filter_gain_done_seen = false;
    ctx->grid_filter_ptnt_done_seen = false;
    ctx->grid_contrast_gain_done_seen = false;
    ctx->grid_contrast_ptnt_done_seen = false;
    ctx->scatter_done_seen = false;
    ctx->passthrough_done_seen = false;
    ctx->median_filter_done_seen = false;
    ctx->correct_done_seen = false;
    ctx->img_width = 0;
    ctx->img_height = 0;
	ctx->pixel_chunk_remaining = 0;
	ctx->pixel_chunk_len = 0;

    ctx->pixel_buf_src = pixel_buf_src;
    ctx->pixel_buf_dst = pixel_buf_dst;
    ctx->fxd_endian = 0;
    ctx->fxd_type_code_in = 0;
    ctx->fxd_frames = 0;
    ctx->fxd_commentlen = 0;

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
    LOG_L2("\nChecking AXI DMA errors...\r\n");

    u32 mm2s = Xil_In32(DMA_BASE + DMA_MM2S_DMASR_OFFSET);
    u32 s2mm = Xil_In32(DMA_BASE + DMA_S2MM_DMASR_OFFSET);

    LOG_L2("MM2S_DMASR = 0x%08x\r\n", mm2s);
    LOG_L2("S2MM_DMASR = 0x%08x\r\n", s2mm);

    if (mm2s & (1<<10))
        LOG_ERR("MM2S ERROR at startup!\r\n");
    else
        LOG_L2("MM2S OK.\r\n");

    if (s2mm & (1<<10))
        LOG_ERR("S2MM ERROR at startup!\r\n");
    else
        LOG_L2("S2MM OK.\r\n");

    LOG_L2("DMA status check complete.\r\n");

    int status;
    XAxiDma_Config *CfgPtr = XAxiDma_LookupConfig(XPAR_AXI_DMA_0_BASEADDR);
    if (CfgPtr == NULL) {
        DEBUG_PRINT("DMA: LookupConfig returned NULL (dev id %d)\r\n", XPAR_AXI_DMA_0_BASEADDR);
        return XST_FAILURE;
    }

    status = XAxiDma_CfgInitialize(&AxiDma, CfgPtr);
    if (status != XST_SUCCESS) {
        DEBUG_PRINT("DMA: CfgInitialize failed: %d\r\n", status);
        return status;
    }


    DEBUG_PRINT("DMA: Initialized OK\r\n");

    /* ---- Print driver-enforced limits/capabilities ----
     * Notes:
     *  - MaxTransferLen comes from the IP parameter “Width of Buffer Length Register”
     *    (LengthWidth). Max bytes per submission = 2^LengthWidth - 1 (PG021). 
     *  - HasDRE==0 means buffer addresses must be aligned to DataWidth bytes for Simple mode.
     *  - TxBdRing fields apply to MM2S (DMA_TO_DEVICE); RxBdRing[0] to S2MM (DEVICE_TO_DMA).
     */
    LOG_L2("DMA caps:\r\n");
    LOG_L2("  HasMm2S=%d HasS2Mm=%d\r\n", AxiDma.HasMm2S, AxiDma.HasS2Mm);

    LOG_L2("  MM2S: MaxTransferLen=%u bytes, DataWidth=%u bytes, HasDRE=%d\r\n",
               AxiDma.TxBdRing.MaxTransferLen,
               AxiDma.TxBdRing.DataWidth,
               AxiDma.TxBdRing.HasDRE);

    LOG_L2("  S2MM: MaxTransferLen=%u bytes, DataWidth=%u bytes, HasDRE=%d\r\n",
               AxiDma.RxBdRing[0].MaxTransferLen,
               AxiDma.RxBdRing[0].DataWidth,
               AxiDma.RxBdRing[0].HasDRE);

    /* Optional: compute a runtime safe chunk size for Simple mode */
    {
        u32 mm2s_limit = AxiDma.TxBdRing.MaxTransferLen;
        u32 s2mm_limit = AxiDma.RxBdRing[0].MaxTransferLen;
        u32 dma_max_xfer = (mm2s_limit < s2mm_limit) ? mm2s_limit : s2mm_limit;
        LOG_L2("  Simple-mode safe chunk size: %u bytes\r\n", dma_max_xfer);
    }

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
