
"""
Diagnose (L0, attach) space: find configs that produce meaningful torque
starting near 0° (early engagement across device ROM -8° to +12°).
"""

import sys
sys.path.insert(0, '/Users/maccamardo/HITLO_Symmetry')

import numpy as np
from hitlo.cost import compute_exo_torque

# Device ROM
ROM_MIN = -8.0  # dorsiflexion limit
ROM_MAX = 12.0  # plantarflexion limit

def evaluate_config(L0, attach):
    """Return (max_pf_torque, max_df_torque, engages_early)."""
    # PF zone: 0° to 12° (device ROM plantarflexion)
    pf_angles = np.linspace(0, ROM_MAX, 50)
    pf_torques = np.array([compute_exo_torque(a, L0, attach) for a in pf_angles])
    
    # DF zone: -8° to 0° (device ROM dorsiflexion)
    df_angles = np.linspace(ROM_MIN, 0, 50)
    df_torques = np.array([compute_exo_torque(a, L0, attach) for a in df_angles])
    
    # Check if torque rises quickly from 0°
    torque_at_0 = abs(compute_exo_torque(0.0, L0, attach))
    torque_at_2 = abs(compute_exo_torque(2.0, L0, attach))
    torque_at_4 = abs(compute_exo_torque(4.0, L0, attach))
    
    max_pf = np.max(pf_torques)
    max_df = np.max(np.abs(df_torques))
    
    engages_early = (torque_at_2 > 2.0)  # Has >2 Nm by 2°
    
    return max_pf, max_df, engages_early, torque_at_0, torque_at_2, torque_at_4

# Scan L0 range, look for PF-assist (negative attach)
print("=" * 80)
print("SCANNING (L0, attach) SPACE FOR EARLY ENGAGEMENT")
print("=" * 80)
print("\nPF-ASSIST (negative attach = -0.15):")
print(f"{'L0 (m)':<10} {'Max PF':<10} {'Max DF':<10} {'@0°':<10} {'@2°':<10} {'@4°':<10} {'Early?':<8}")
print("-" * 80)

attach = -0.15
for L0 in np.linspace(0.30, 0.42, 25):
    max_pf, max_df, early, t0, t2, t4 = evaluate_config(L0, attach)
    early_str = "✓" if early else "✗"
    print(f"{L0:.4f}      {max_pf:+8.2f}   {max_df:+8.2f}   {t0:+8.2f}   {t2:+8.2f}   {t4:+8.2f}   {early_str}")

print("\nDF-ASSIST (positive attach = 0.5):")
print(f"{'L0 (m)':<10} {'Max PF':<10} {'Max DF':<10} {'@0°':<10} {'@-2°':<10} {'@-4°':<10} {'Early?':<8}")
print("-" * 80)

attach = 0.5
for L0 in np.linspace(0.30, 0.42, 25):
    max_pf, max_df, early, t0, t2, t4 = evaluate_config(L0, attach)
    # For DF-assist, check negative angles
    torque_at_neg2 = abs(compute_exo_torque(-2.0, L0, attach))
    torque_at_neg4 = abs(compute_exo_torque(-4.0, L0, attach))
    early_str = "✓" if torque_at_neg2 > 2.0 else "✗"
    print(f"{L0:.4f}      {max_pf:+8.2f}   {max_df:+8.2f}   {t0:+8.2f}   {torque_at_neg2:+8.2f}   {torque_at_neg4:+8.2f}   {early_str}")

print("\n" + "=" * 80)
print("INTERPRETATION:")
print("  - L0 controls ENGAGEMENT TIMING (lower L0 = earlier engagement)")
print("  - attach controls DIRECTION (negative = PF, positive = DF)")
print("  - Look for configs where |torque| > 2 Nm by ±2° from neutral")
print("=" * 80)