"""Shared marker / colour / line style of the two cost-model figures, so they read as one family:
  s12_512_constrained_estimated.png  (Dice vs modelled resource %, plot_constrained_builds.py)
  s12_512_cost_model_error.png       (model error vs synthesis, plot_model_error.py)
Shape + colour = resource (blue circle LUT, orange square BRAM18K, aqua triangle DSP); fill = estimate (hollow calibrated, filled uncalibrated); line dashed = calibrated, solid = uncalibrated.
Colours: slots 1-3 of the reference categorical palette (blue, orange, aqua), validated all-pairs; aqua is below 3:1 on the light surface, so identity also rides on the marker shape and the legend."""

MARK_SIZE, MARK_WIDTH = 4.5, 0.9         # small, thin-edged shapes
ALPHA = 0.85                             # markers
LINE_ALPHA = 0.85                        # connecting lines
LINE_WIDTH = 1.0
CAL_STYLE, UNCAL_STYLE = "--", "-"       # calibrated dashed, uncalibrated solid
GUIDE_COLOR = "#c3c2b7"                  # faint vertical guides, legend frame
# resource -> (marker, legend label, colour)
RESOURCE_STYLE = {
    "lut": ("o", "LUT", "#2a78d6"),
    "bram": ("s", "BRAM18K", "#eb6834"),
    "dsp": ("^", "DSP", "#1baf7a"),
}
