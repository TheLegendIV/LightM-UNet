# 
# Usage: To re-create this platform project launch xsct with below options.
# xsct C:\DEV\repos\LightM-UNet\deployment\FINN_interface\zcu106_test\platform.tcl
# 
# OR launch xsct and run below command.
# source C:\DEV\repos\LightM-UNet\deployment\FINN_interface\zcu106_test\platform.tcl
# 
# To create the platform in a different location, modify the -out option of "platform create" command.
# -out option specifies the output directory of the platform project.

platform create -name {zcu106_test}\
-hw {C:\DEV\repos\LightM-UNet\deployment\top_wrapper.xsa}\
-proc {psu_cortexa53_0} -os {standalone} -arch {64-bit} -fsbl-target {psu_cortexa53_0} -out {C:/DEV/repos/LightM-UNet/deployment/FINN_interface}

platform write
platform generate -domains 
domain active {zynqmp_pmufw}
bsp reload
domain active {standalone_domain}
bsp reload
bsp setlib -name lwip211 -ver 1.8
bsp setlib -name xiltimer -ver 1.1
bsp write
bsp reload
catch {bsp regenerate}
platform generate
