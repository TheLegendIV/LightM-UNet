
// sys_time.c (patched)
#include "lwip/opt.h"
#include "lwip/sys.h"
#include "xparameters.h"
#include "xil_types.h"
#include "xttcps.h"

static XTtcPs TtcInst;
static volatile u32_t ms_ticks = 0;

static int TtcPs_Init_1ms_any(void)
{
    int dev_id = -1;
    XTtcPs_Config *cfg = NULL;

    for (int i = 0; i < 12; ++i) {
        cfg = XTtcPs_LookupConfig(i);
        if (cfg) { dev_id = i; break; }
    }
    if (dev_id < 0) return -1;

    if (XTtcPs_CfgInitialize(&TtcInst, cfg, cfg->BaseAddress) != XST_SUCCESS)
        return -2;

    XTtcPs_SetOptions(&TtcInst,
        XTTCPS_OPTION_INTERVAL_MODE | XTTCPS_OPTION_WAVE_DISABLE);

    // Compute 1 ms interval
    u32 interval, prescaler;
    XTtcPs_CalcIntervalFromFreq(&TtcInst, 1000, &interval, &prescaler);
    XTtcPs_SetPrescaler(&TtcInst, prescaler);
    XTtcPs_SetInterval(&TtcInst, interval);

    // *** Enable interval event so GetInterruptStatus reports it ***
    XTtcPs_ClearInterruptStatus(&TtcInst, XTTCPS_IXR_INTERVAL_MASK);
    XTtcPs_EnableInterrupts(&TtcInst, XTTCPS_IXR_INTERVAL_MASK);

    XTtcPs_Start(&TtcInst);
    return dev_id;
}

static inline void TtcPs_PollTick(void)
{
    u32 status = XTtcPs_GetInterruptStatus(&TtcInst);
    if (status & XTTCPS_IXR_INTERVAL_MASK) {
        ms_ticks++;
        XTtcPs_ClearInterruptStatus(&TtcInst, XTTCPS_IXR_INTERVAL_MASK);
    }
}

void sys_time_init(void) { (void)TtcPs_Init_1ms_any(); }
void sys_time_poll(void) { TtcPs_PollTick(); }
u32_t sys_now(void) { return ms_ticks; }
