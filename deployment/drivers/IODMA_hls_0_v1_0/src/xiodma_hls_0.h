// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef XIODMA_HLS_0_H
#define XIODMA_HLS_0_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xiodma_hls_0_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
    u16 DeviceId;
    u64 Control_BaseAddress;
} XIodma_hls_0_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XIodma_hls_0;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XIodma_hls_0_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XIodma_hls_0_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XIodma_hls_0_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XIodma_hls_0_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
int XIodma_hls_0_Initialize(XIodma_hls_0 *InstancePtr, u16 DeviceId);
XIodma_hls_0_Config* XIodma_hls_0_LookupConfig(u16 DeviceId);
int XIodma_hls_0_CfgInitialize(XIodma_hls_0 *InstancePtr, XIodma_hls_0_Config *ConfigPtr);
#else
int XIodma_hls_0_Initialize(XIodma_hls_0 *InstancePtr, const char* InstanceName);
int XIodma_hls_0_Release(XIodma_hls_0 *InstancePtr);
#endif

void XIodma_hls_0_Start(XIodma_hls_0 *InstancePtr);
u32 XIodma_hls_0_IsDone(XIodma_hls_0 *InstancePtr);
u32 XIodma_hls_0_IsIdle(XIodma_hls_0 *InstancePtr);
u32 XIodma_hls_0_IsReady(XIodma_hls_0 *InstancePtr);
void XIodma_hls_0_EnableAutoRestart(XIodma_hls_0 *InstancePtr);
void XIodma_hls_0_DisableAutoRestart(XIodma_hls_0 *InstancePtr);

void XIodma_hls_0_Set_out_V(XIodma_hls_0 *InstancePtr, u64 Data);
u64 XIodma_hls_0_Get_out_V(XIodma_hls_0 *InstancePtr);
void XIodma_hls_0_Set_numReps(XIodma_hls_0 *InstancePtr, u32 Data);
u32 XIodma_hls_0_Get_numReps(XIodma_hls_0 *InstancePtr);

void XIodma_hls_0_InterruptGlobalEnable(XIodma_hls_0 *InstancePtr);
void XIodma_hls_0_InterruptGlobalDisable(XIodma_hls_0 *InstancePtr);
void XIodma_hls_0_InterruptEnable(XIodma_hls_0 *InstancePtr, u32 Mask);
void XIodma_hls_0_InterruptDisable(XIodma_hls_0 *InstancePtr, u32 Mask);
void XIodma_hls_0_InterruptClear(XIodma_hls_0 *InstancePtr, u32 Mask);
u32 XIodma_hls_0_InterruptGetEnabled(XIodma_hls_0 *InstancePtr);
u32 XIodma_hls_0_InterruptGetStatus(XIodma_hls_0 *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
