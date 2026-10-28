#!/bin/bash
f=$(find / -name 'upsample.hpp' 2>/dev/null | head -1)
echo "$f"
grep -n -B2 -A55 'void UpsampleNearestNeighbour_Batch' "$f" | head -100
