set -e
P=/home/thelegendiv/tools/verilator5
mkdir -p /home/thelegendiv/tools && cd /home/thelegendiv/tools
rm -rf verilator_src
git clone https://github.com/verilator/verilator verilator_src
cd verilator_src
TAG=$(git tag -l 'v5.0[0-9][0-9]' | sort -V | tail -1)
echo "using $TAG"
git checkout -q $TAG
autoconf
./configure --prefix=$P
make -j10
make install
$P/bin/verilator --version
