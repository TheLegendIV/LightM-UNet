// Bare-metal helper utilities for grid_filter coefficient preparation.
// - No filesystem access, no STL, no HLS C-sim dependencies.
// - Uses an embedded float kernel matrix (generated from Grid-filter-kernels.txt).

#include "gsc_utils.h"

#include "grid_filter_kernels_float.h"

#include <math.h>
static long gsc_lroundf(float x) { return lroundf(x); }

#ifndef GSC_GAUSS_KERNEL_MAX_TAPS
#define GSC_GAUSS_KERNEL_MAX_TAPS (63u)
#endif

#if defined(GSC_KERNEL_TAPS) && (GSC_KERNEL_TAPS != 35)
#error "Unexpected GSC_KERNEL_TAPS; update gsc_utils.h defaults"
#endif

static int16_t gsc_sat_int16_from_int(int v) {
    if (v > 32767) return (int16_t)32767;
    if (v < -32768) return (int16_t)-32768;
    return (int16_t)v;
}

static int16_t gsc_quantize_q(float v, int frac_bits) {
    const float scaled = v * (float)(1 << frac_bits);
    const long q = gsc_lroundf(scaled);
    return gsc_sat_int16_from_int((int)q);
}

static bool gsc_normalize_float_inplace(float* taps, size_t n) {
    if (!taps || n == 0u) return false;

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += (double)taps[i];
    if (sum == 0.0) return false;

    const float inv_sum = (float)(1.0 / sum);
    for (size_t i = 0; i < n; ++i) taps[i] *= inv_sum;
    return true;
}

static int gsc_make_odd_int(int x) {
    if (x < 1) x = 1;
    if ((x & 1) == 0) ++x;
    return x;
}

static bool gsc_make_gauss_kernel_1d_float(float* kernel, size_t n, double sigma) {
    if (!kernel || n == 0u) return false;
    if ((n & 1u) == 0u) return false;

    const int half = (int)(n / 2u);
    double sum = 0.0;
    for (int i = -half; i <= half; ++i) {
        double f = (sigma > 0.0) ? ((double)i / sigma) : 1.0;
        f = exp(-0.5 * f * f);
        kernel[i + half] = (float)f;
        sum += (double)kernel[i + half];
    }

    if (sum == 0.0) {
        for (size_t i = 0; i < n; ++i) kernel[i] = 0.0f;
        return false;
    }
    const float inv_sum = (float)(1.0 / sum);
    for (size_t i = 0; i < n; ++i) kernel[i] *= inv_sum;
    return true;
}

static void gsc_balance_unit_sum_q_inplace(int16_t* taps_q, size_t n, int frac_bits) {
    if (!taps_q || n == 0u) return;

    const int target = (1 << frac_bits);

    int sum_before = 0;
    for (size_t i = 0; i < n; ++i) sum_before += (int)taps_q[i];

    const int delta = target - sum_before;
    const int base = delta / (int)n;
    int rem = delta - (base * (int)n);

    for (size_t i = 0; i < n; ++i) {
        taps_q[i] = gsc_sat_int16_from_int((int)taps_q[i] + base);
    }

    if (rem != 0) {
        const int step = (rem > 0) ? 1 : -1;
        int absrem = (rem > 0) ? rem : -rem;
        const int center = (int)(n / 2u);

        if (absrem > 0) {
            taps_q[center] = gsc_sat_int16_from_int((int)taps_q[center] + step);
            --absrem;
        }

        for (int k = 1; absrem > 0; ++k) {
            const int li = center - k;
            const int ri = center + k;
            if (li < 0 || ri >= (int)n) break;
            if (absrem >= 2) {
                taps_q[li] = gsc_sat_int16_from_int((int)taps_q[li] + step);
                taps_q[ri] = gsc_sat_int16_from_int((int)taps_q[ri] + step);
                absrem -= 2;
            } else {
                taps_q[center] = gsc_sat_int16_from_int((int)taps_q[center] + step);
                absrem = 0;
            }
        }
    }
}

static void gsc_make_boxcar_q_exact(int16_t* taps_q_out, size_t n, int frac_bits) {
    if (!taps_q_out || n == 0u) return;

    const int target = (1 << frac_bits);
    const int q0 = target / (int)n;
    const int r = target - (q0 * (int)n);
    const int q1 = q0 + 1;

    for (size_t i = 0; i < n; ++i) taps_q_out[i] = gsc_sat_int16_from_int(q0);

    const int center = (int)(n / 2u);
    int remaining = r;
    if ((remaining & 1) && (center >= 0) && (center < (int)n)) {
        taps_q_out[center] = gsc_sat_int16_from_int(q1);
        --remaining;
    }
    for (int k = 1; remaining > 0; ++k) {
        const int li = center - k;
        const int ri = center + k;
        if (li >= 0 && ri < (int)n) {
            taps_q_out[li] = gsc_sat_int16_from_int(q1);
            taps_q_out[ri] = gsc_sat_int16_from_int(q1);
            remaining -= 2;
        } else {
            break;
        }
    }
}

int gsc_grid_filter_nr_from_frequency(float grid_frequency) {
    int filter_NR = 0;
    if ((grid_frequency >= 40.000f) && (grid_frequency < 40.750f)) filter_NR = 1;
    if ((grid_frequency >= 40.750f) && (grid_frequency < 41.250f)) filter_NR = 2;
    if ((grid_frequency >= 41.250f) && (grid_frequency < 41.750f)) filter_NR = 3;
    if ((grid_frequency >= 41.750f) && (grid_frequency < 42.250f)) filter_NR = 4;
    if ((grid_frequency >= 42.250f) && (grid_frequency < 42.750f)) filter_NR = 5;
    if ((grid_frequency >= 42.750f) && (grid_frequency < 43.250f)) filter_NR = 6;
    if ((grid_frequency >= 43.250f) && (grid_frequency < 43.750f)) filter_NR = 7;
    if ((grid_frequency >= 43.750f) && (grid_frequency < 44.250f)) filter_NR = 8;
    if ((grid_frequency >= 44.250f) && (grid_frequency < 44.750f)) filter_NR = 9;
    if ((grid_frequency >= 44.750f) && (grid_frequency < 45.250f)) filter_NR = 10;
    if ((grid_frequency >= 45.250f) && (grid_frequency < 46.000f)) filter_NR = 11;
    return filter_NR;
}

bool gsc_get_grid_filter_y_taps_float(float grid_frequency,
                                     unsigned vertical_binning_1based,
                                     float* y_taps_out,
                                     size_t y_taps_out_count) {
    if (!y_taps_out || y_taps_out_count < (size_t)GSC_KERNEL_TAPS) return false;

    const int filter_NR = gsc_grid_filter_nr_from_frequency(grid_frequency);
    if (filter_NR < 1 || filter_NR > (int)GSC_KERNEL_FILTERS) return false;
    if (vertical_binning_1based < 1u || vertical_binning_1based > 4u) return false;

    const size_t col = gsc_kernel_col_index((unsigned)filter_NR, vertical_binning_1based);
    if (col >= (size_t)GSC_KERNEL_COLS) return false;

    for (size_t i = 0; i < (size_t)GSC_KERNEL_TAPS; ++i) {
        y_taps_out[i] = g_grid_filter_kernels[col][i];
    }

    return gsc_normalize_float_inplace(y_taps_out, (size_t)GSC_KERNEL_TAPS);
}

bool gsc_get_grid_filter_y_taps_q(int frac_bits,
                                 float grid_frequency,
                                 unsigned vertical_binning_1based,
                                 int16_t* y_taps_q_out,
                                 size_t y_taps_q_out_count) {
    if (!y_taps_q_out || y_taps_q_out_count < (size_t)GSC_KERNEL_TAPS) return false;
    if (frac_bits < 0 || frac_bits > 15) return false;

    float y_taps[(size_t)GSC_KERNEL_TAPS];
    if (!gsc_get_grid_filter_y_taps_float(grid_frequency, vertical_binning_1based, y_taps, (size_t)GSC_KERNEL_TAPS)) {
        return false;
    }

    for (size_t i = 0; i < (size_t)GSC_KERNEL_TAPS; ++i) {
        y_taps_q_out[i] = gsc_quantize_q(y_taps[i], frac_bits);
    }

#if defined(GRID_FILTER_BALANCE_TAPS) && (GRID_FILTER_BALANCE_TAPS != 0)
    gsc_balance_unit_sum_q_inplace(y_taps_q_out, (size_t)GSC_KERNEL_TAPS, frac_bits);
#endif

    return true;
}

void gsc_make_box_filter_x_taps_float(float* x_taps_out, size_t n) {
    if (!x_taps_out || n == 0u) return;

    const float v = 1.0f / (float)n;
    for (size_t i = 0; i < n; ++i) x_taps_out[i] = v;
}

bool gsc_make_box_filter_x_taps_q(int frac_bits, int16_t* x_taps_q_out, size_t n) {
    if (!x_taps_q_out || n == 0u) return false;
    if (frac_bits < 0 || frac_bits > 15) return false;

#if defined(GRID_FILTER_BALANCE_TAPS) && (GRID_FILTER_BALANCE_TAPS != 0)
    gsc_make_boxcar_q_exact(x_taps_q_out, n, frac_bits);
#else
    // Uniform rounded taps (may drift slightly from exact unit sum).
    const int q = (int)gsc_lroundf((float)(1 << frac_bits) / (float)n);
    for (size_t i = 0; i < n; ++i) x_taps_q_out[i] = gsc_sat_int16_from_int(q);
#endif

    return true;
}

bool gsc_make_gauss_kernel_1d_q(int frac_bits,
                               float sigma,
                               float truncation,
                               int16_t* taps_q_out,
                               size_t taps_q_out_count) {
    if (!taps_q_out) return false;
    if (frac_bits < 0 || frac_bits > 15) return false;
    if (sigma < 0.0f || truncation < 0.0f) return false;

    const double expected_size_f = ceil((double)sigma * (double)truncation * 2.0);
    const int expected_size = gsc_make_odd_int((int)expected_size_f);
    if ((size_t)expected_size != taps_q_out_count) return false;
    if (taps_q_out_count > (size_t)GSC_GAUSS_KERNEL_MAX_TAPS) return false;
    if ((taps_q_out_count & 1u) == 0u) return false;

    float taps_f[GSC_GAUSS_KERNEL_MAX_TAPS];
    if (!gsc_make_gauss_kernel_1d_float(taps_f, taps_q_out_count, (double)sigma)) {
        return false;
    }

    for (size_t i = 0; i < taps_q_out_count; ++i) {
        taps_q_out[i] = gsc_quantize_q(taps_f[i], frac_bits);
    }

    // Force unit-sum in Q before packing/programming.
    gsc_balance_unit_sum_q_inplace(taps_q_out, taps_q_out_count, frac_bits);
    return true;
}
