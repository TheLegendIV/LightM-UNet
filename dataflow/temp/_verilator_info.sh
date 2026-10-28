cd /home/thelegendiv/finn
git grep -n -i "verilator" FETCH_HEAD -- docker/Dockerfile.finn requirements.txt setup.cfg docker/finn_entrypoint.sh | head -20
echo --- local
grep -n -i "verilator" docker/Dockerfile.finn | head
echo --- who uses pyverilator in the local build flow
grep -rln "pyverilator\|PyVerilator" src/finn --include=*.py | head -20
echo --- docker/can I build
which gcc g++ make flex bison autoconf git; nproc
df -h /tmp | tail -1
