# 
# Usage: To re-create this platform project launch xsct with below options.
# xsct C:\DEV\repos\LightM-UNet\deployment\FINN_interface\zcu106\platform.tcl
# 
# OR launch xsct and run below command.
# source C:\DEV\repos\LightM-UNet\deployment\FINN_interface\zcu106\platform.tcl
# 
# To create the platform in a different location, modify the -out option of "platform create" command.
# -out option specifies the output directory of the platform project.

platform create -name {zcu106}\
-hw {F:\PLD\Xilinx\Vitis\2022.2\data\embeddedsw\lib\fixed_hwplatforms\zcu106.xsa}\
-proc {psu_cortexa53_0} -os {standalone} -arch {64-bit} -fsbl-target {psu_cortexa53_0} -out {C:/DEV/repos/LightM-UNet/deployment/FINN_interface}

platform write
platform generate -domains 
platform active {zcu106}
domain active {zynqmp_fsbl}
bsp reload
domain active {standalone_domain}
bsp reload
platform clean
platform generate
