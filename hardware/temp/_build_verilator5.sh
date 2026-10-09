set -e
P=/home/thelegendiv/tools/verilator5
mkdir -p /home/thelegendiv/tools && cd /home/thelegendiv/tools
if [ ! -d verilator_src ]; then git clone --depth 50 https://github.com/verilator/verilator verilator_src; fi
cd verilator_src
git fetch --tags -q || true
git checkout -q stable
git log -1 --format='%h %ad %s' --date=short
autoconf
./configure --prefix=$P
make -j10
make install
$P/bin/verilator --version
