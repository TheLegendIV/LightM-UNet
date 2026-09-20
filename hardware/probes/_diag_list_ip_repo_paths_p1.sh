#!/bin/bash
grep -oP '(?<=<Option Name="IPRepoPath" Val=")[^"]+' /tmp/finn_dev_thelegendiv/vivado_stitch_proj_rtj421tm/finn_vivado_stitch_proj.xpr
