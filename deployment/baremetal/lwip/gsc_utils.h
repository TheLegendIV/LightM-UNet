#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Defaults for grid_filter kernels (current design: 35 Y taps, 21 X taps).
#ifndef GSC_Q_FRAC_BITS
#define GSC_Q_FRAC_BITS (14)
#endif

// Balancing makes sum(taps_q) == (1<<frac_bits) exactly/near-exact.
#ifndef GRID_FILTER_BALANCE_TAPS
#define GRID_FILTER_BALANCE_TAPS (1)
#endif

// Returns 1..11 if in range, else 0.
int gsc_grid_filter_nr_from_frequency(float grid_frequency);

// Extracts + normalizes 35 float Y taps for a given frequency/binning.
// vertical_binning_1based: 1=1x1, 2=2x2, 3=2x4, 4=4x4.
// y_taps_out must have at least 35 entries.
bool gsc_get_grid_filter_y_taps_float(float grid_frequency,
                                     unsigned vertical_binning_1based,
                                     float* y_taps_out,
                                     size_t y_taps_out_count);

// Extracts + normalizes + quantizes Y taps into signed Q format (default Q14).
// When GRID_FILTER_BALANCE_TAPS!=0, adjusts LSBs so sum == (1<<frac_bits).
// y_taps_q_out must have at least 35 entries.
bool gsc_get_grid_filter_y_taps_q(int frac_bits,
                                 float grid_frequency,
                                 unsigned vertical_binning_1based,
                                 int16_t* y_taps_q_out,
                                 size_t y_taps_q_out_count);

// Generates X taps as a box filter.
// - Float: each tap == 1/n (sum == 1)
// - Q: when GRID_FILTER_BALANCE_TAPS!=0, uses exact unit-sum construction.
void gsc_make_box_filter_x_taps_float(float* x_taps_out, size_t n);
bool gsc_make_box_filter_x_taps_q(int frac_bits, int16_t* x_taps_q_out, size_t n);

// Generates a 1D Gaussian kernel using the same sizing rule as the SW helper:
// kernelsize = makeOdd(ceil(sigma * truncation * 2.0)).
// - Produces float-normalized taps (sum == 1.0), then quantizes to signed Q format.
// - Applies unit-sum balancing in Q so sum(taps_q) == (1<<frac_bits) (exact/near-exact).
// Returns false on parameter mismatch (e.g. out_count does not match computed kernelsize).
bool gsc_make_gauss_kernel_1d_q(int frac_bits,
                               float sigma,
                               float truncation,
                               int16_t* taps_q_out,
                               size_t taps_q_out_count);
