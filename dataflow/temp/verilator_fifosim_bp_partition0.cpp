/* Based on FINN's stock verilator_fifosim.cpp testbench (same license),
modified to drive an AXI-stream-correct, optionally backpressuring
m_axis_0_tready on the OUTPUT side -- the stock driver hardcodes
m_axis_0_tready=1 permanently right after reset, so it can never exercise
(or catch a bug in) how the design behaves when its downstream consumer
stalls. This variant targets partition 0's OWN standalone stitched-IP
top module (GenericPartition_0_wrapper), to test in isolation whether it
can deadlock under realistic output backpressure -- matching real
hardware ILA evidence that partition 0 absorbs input but never reasserts
m_axis_0_tvalid once something downstream stops being ready. */

#include <iostream>
#include <fstream>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include "verilated.h"
#include "VGenericPartition_0_wrapper.h"

using namespace std;

VGenericPartition_0_wrapper * top;
double main_time = 0;
double sc_time_stamp() { return main_time; }

VGenericPartition_0_wrapper* construct() {
    Verilated::commandArgs(0, (const char**) nullptr);
    VGenericPartition_0_wrapper* top = new VGenericPartition_0_wrapper();
    return top;
}
void destruct(VGenericPartition_0_wrapper* top) {
    if (top != nullptr) { delete top; }
}

inline void toggle_clk() {
    top->eval(); main_time++;
    top->ap_clk = 1;
    top->eval(); main_time++;
    top->ap_clk = 0;
}

void reset() {
    top->ap_rst_n = 0;
    for (unsigned i = 0; i < 10; i++) toggle_clk();
    top->ap_rst_n = 1;
}

int main(int argc, char *argv[]) {
    bool backpressure = (argc > 1) && (strcmp(argv[1], "bp") == 0);
    unsigned bp_ready_len = (argc > 2) ? (unsigned)atoi(argv[2]) : 16;
    unsigned bp_stall_len = (argc > 3) ? (unsigned)atoi(argv[3]) : 16;
    unsigned max_iters = (argc > 4) ? (unsigned)atoi(argv[4]) : 5000000;

    top = construct();

    unsigned n_iters_per_input = 65536;
    unsigned n_iters_per_output = 65536;
    unsigned n_inputs = 1;

    reset();

    top->s_axis_0_tvalid = 1;
    top->m_axis_0_tready = backpressure ? 0 : 1;

    unsigned n_in_txns = 0, n_out_txns = 0, iters = 0, last_output_at = 0;
    unsigned latency = 0;
    unsigned bp_phase_counter = 0;
    bool bp_ready_phase = true;
    bool exit_criterion = false;

    cout << "Simulation starting (backpressure=" << (backpressure ? "ON" : "OFF")
         << " ready_len=" << bp_ready_len << " stall_len=" << bp_stall_len
         << " max_iters=" << max_iters << ")" << endl;
    cout << "Number of inputs to write " << n_iters_per_input * n_inputs << endl;
    cout << "Number of outputs to expect " << n_iters_per_output * n_inputs << endl;

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    while (!exit_criterion) {
        toggle_clk();
        iters++;

        if (backpressure) {
            bp_phase_counter++;
            unsigned limit = bp_ready_phase ? bp_ready_len : bp_stall_len;
            if (bp_phase_counter >= limit) {
                bp_phase_counter = 0;
                bp_ready_phase = !bp_ready_phase;
            }
            top->m_axis_0_tready = bp_ready_phase ? 1 : 0;
        }

        if (iters % 1000 == 0) {
            cout << "Elapsed iters " << iters << " inps " << n_in_txns << " outs " << n_out_txns << endl;
            chrono::steady_clock::time_point end = chrono::steady_clock::now();
            cout << "Elapsed since last report = "
                 << chrono::duration_cast<chrono::seconds>(end - begin).count() << "[s]" << endl;
            begin = end;
        }

        if (top->s_axis_0_tready == 1 && top->s_axis_0_tvalid == 1) {
            n_in_txns++;
            if (n_in_txns == n_iters_per_input * n_inputs) {
                top->s_axis_0_tvalid = 0;
                cout << "All inputs written at cycle " << iters << endl;
            }
        }
        if (top->m_axis_0_tvalid == 1 && top->m_axis_0_tready == 1) {
            n_out_txns++;
            last_output_at = iters;
            if (n_out_txns == n_iters_per_output) latency = iters;
        }

        exit_criterion = ((n_in_txns >= n_iters_per_input * n_inputs) &&
                           (n_out_txns >= n_iters_per_output * n_inputs)) ||
                          ((iters - last_output_at) > max_iters);
    }

    cout << "Simulation finished" << endl;
    cout << "Number of inputs consumed " << n_in_txns << endl;
    cout << "Number of outputs produced " << n_out_txns << endl;
    cout << "Number of clock cycles " << iters << endl;

    ofstream results_file;
    results_file.open("results.txt", ios::out | ios::trunc);
    results_file << "N_IN_TXNS" << "\t" << n_in_txns << endl;
    results_file << "N_OUT_TXNS" << "\t" << n_out_txns << endl;
    results_file << "cycles" << "\t" << iters << endl;
    results_file << "N" << "\t" << n_inputs << endl;
    results_file << "latency_cycles" << "\t" << latency << endl;
    results_file.close();

    destruct(top);
    return 0;
}
