// Standalone randomised-stall testbench for the P7 MVAU_rtl_1 wrapper (MW=1 MH=4 PE=1 SIMD=1, INT6 w / UINT6 act).
// args: NPIX p_in_valid p_w_valid p_out_ready seed [first_act_lo first_act_hi]
#include "VGenericPartition_7_MVAU_rtl_1.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

static int sx13(unsigned v) { v &= 0x1fff; return (v & 0x1000) ? (int)v - 0x2000 : (int)v; }

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    int NPIX = atoi(argv[1]), pin = atoi(argv[2]), pw = atoi(argv[3]), pout = atoi(argv[4]);
    unsigned seed = atoi(argv[5]);
    int lo = argc > 6 ? atoi(argv[6]) : 0, hi = argc > 7 ? atoi(argv[7]) : 63;
    std::mt19937 rng(seed);
    auto chance = [&](int p) { return (int)(rng() % 100) < p; };
    const int W[4] = {27, 31, -15, 10};
    auto *t = new VGenericPartition_7_MVAU_rtl_1;
    std::vector<int> acts(NPIX);
    for (auto &a : acts) a = lo + rng() % (hi - lo + 1);
    std::vector<int> got;
    int ai = 0, wptr = 0;
    bool in_v = false, w_v = false;
    long cyc = 0, last_prog = 0;
    t->ap_rst_n = 0; t->in0_V_TVALID = 0; t->weights_V_TVALID = 0; t->out_V_TREADY = 0;
    for (int i = 0; i < 10; i++) { t->ap_clk = 0; t->eval(); t->ap_clk = 1; t->eval(); }
    t->ap_rst_n = 1;
    while ((int)got.size() < NPIX * 4 && cyc - last_prog < 20000) {
        // decide stimulus (hold until accepted)
        if (!in_v && ai < NPIX && chance(pin)) in_v = true;
        if (!w_v && chance(pw)) w_v = true;
        t->in0_V_TVALID = in_v; t->in0_V_TDATA = acts[ai < NPIX ? ai : 0];
        t->weights_V_TVALID = w_v; t->weights_V_TDATA = W[wptr] & 0x3f;
        bool oready = chance(pout);
        t->out_V_TREADY = oready;
        t->ap_clk = 0; t->eval();
        bool in_acc = in_v && t->in0_V_TREADY, w_acc = w_v && t->weights_V_TREADY;
        bool o_acc = t->out_V_TVALID && oready;
        if (o_acc) { got.push_back(sx13(t->out_V_TDATA)); last_prog = cyc; }
        t->ap_clk = 1; t->eval();
        if (in_acc) { ai++; in_v = false; last_prog = cyc; }
        if (w_acc) { wptr = (wptr + 1) & 3; w_v = false; }
        cyc++;
    }
    int bad = 0, firstbad = -1;
    for (size_t i = 0; i < got.size(); i++) {
        int exp = acts[i / 4] * W[i & 3];
        if (got[i] != exp) { if (firstbad < 0) firstbad = i; bad++; }
    }
    printf("NPIX %d pin %d pw %d pout %d seed %u act[%d,%d]: outputs %zu/%d bad %d firstbad %d cycles %ld\n", NPIX, pin, pw, pout, seed, lo, hi, got.size(), NPIX * 4, bad, firstbad, cyc);
    if (firstbad >= 0) {
        int i = firstbad;
        printf("  first bad elem %d (pix %d ch %d): got %d exp %d (x=%d)\n", i, i / 4, i & 3, got[i], acts[i / 4] * W[i & 3], acts[i / 4]);
    }
    return bad != 0;
}
