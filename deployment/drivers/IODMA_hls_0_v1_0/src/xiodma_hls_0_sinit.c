// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#include "xparameters.h"
#include "xiodma_hls_0.h"

extern XIodma_hls_0_Config XIodma_hls_0_ConfigTable[];

XIodma_hls_0_Config *XIodma_hls_0_LookupConfig(u16 DeviceId) {
	XIodma_hls_0_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XIODMA_HLS_0_NUM_INSTANCES; Index++) {
		if (XIodma_hls_0_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XIodma_hls_0_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XIodma_hls_0_Initialize(XIodma_hls_0 *InstancePtr, u16 DeviceId) {
	XIodma_hls_0_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XIodma_hls_0_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XIodma_hls_0_CfgInitialize(InstancePtr, ConfigPtr);
}

#endif

