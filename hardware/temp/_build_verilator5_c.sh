set -e
cd /home/thelegendiv/tools/verilator_src
grep -n "^HELP2MAN" Makefile | head -3
make -j10 HELP2MAN=true
make install HELP2MAN=true
/home/thelegendiv/tools/verilator5/bin/verilator --version
