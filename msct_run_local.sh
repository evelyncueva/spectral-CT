#!/bin/bash

conda activate odl-py35

echo Experiment 0

# 0: name,    1: recons_size, 2: views_angles,    3: views_det, 4: eta,
# 5: alpha,   6: ref_type,    7: sinfo_alpha_str, 8: algoritms, 9: energies

python msct_main.py geometric 512 90 552 1e-2 5e-2 TV 0_0001 bregmandTV E2










