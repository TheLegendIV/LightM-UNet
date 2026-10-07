// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
/***************************** Include Files *********************************/
#include "xiodma_hls_0.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XIodma_hls_0_CfgInitialize(XIodma_hls_0 *InstancePtr, XIodma_hls_0_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XIodma_hls_0_Start(XIodma_hls_0 *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XIodma_hls_0_ReadReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_AP_CTRL) & 0x80;
    XIodma_hls_0_WriteReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XIodma_hls_0_IsDone(XIodma_hls_0 *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XIodma_hls_0_ReadReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XIodma_hls_0_IsIdle(XIodma_hls_0 *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XIodma_hls_0_ReadReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XIodma_hls_0_IsReady(XIodma_hls_0 *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XIodma_hls_0_ReadReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XIodma_hls_0_EnableAutoRestart(XIodma_hls_0 *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XIodma_hls_0_WriteReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XIodma_hls_0_DisableAutoRestart(XIodma_hls_0 *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XIodma_hls_0_WriteReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_AP_CTRL, 0);
}

void XIodma_hls_0_Set_out_V(XIodma_hls_0 *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XIodma_hls_0_WriteReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_OUT_V_DATA, (u32)(Data));
    XIodma_hls_0_WriteReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_OUT_V_DATA + 4, (u32)(Data >> 32));
}

u64 XIodma_hls_0_Get_out_V(XIodma_hls_0 *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XIodma_hls_0_ReadReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_OUT_V_DATA);
    Data += (u64)XIodma_hls_0_ReadReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_OUT_V_DATA + 4) << 32;
    return Data;
}

void XIodma_hls_0_Set_numReps(XIodma_hls_0 *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XIodma_hls_0_WriteReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_NUMREPS_DATA, Data);
}

u32 XIodma_hls_0_Get_numReps(XIodma_hls_0 *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XIodma_hls_0_ReadReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_NUMREPS_DATA);
    return Data;
}

void XIodma_hls_0_InterruptGlobalEnable(XIodma_hls_0 *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XIodma_hls_0_WriteReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_GIE, 1);
}

void XIodma_hls_0_InterruptGlobalDisable(XIodma_hls_0 *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XIodma_hls_0_WriteReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_GIE, 0);
}

void XIodma_hls_0_InterruptEnable(XIodma_hls_0 *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XIodma_hls_0_ReadReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_IER);
    XIodma_hls_0_WriteReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_IER, Register | Mask);
}

void XIodma_hls_0_InterruptDisable(XIodma_hls_0 *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XIodma_hls_0_ReadReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_IER);
    XIodma_hls_0_WriteReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_IER, Register & (~Mask));
}

void XIodma_hls_0_InterruptClear(XIodma_hls_0 *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XIodma_hls_0_WriteReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_ISR, Mask);
}

u32 XIodma_hls_0_InterruptGetEnabled(XIodma_hls_0 *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XIodma_hls_0_ReadReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_IER);
}

u32 XIodma_hls_0_InterruptGetStatus(XIodma_hls_0 *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XIodma_hls_0_ReadReg(InstancePtr->Control_BaseAddress, XIODMA_HLS_0_CONTROL_ADDR_ISR);
}

