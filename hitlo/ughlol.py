import sys
sys.path.insert(0, '/Users/maccamardo/HITLO_Symmetry')

import numpy as np
import matplotlib.pyplot as plt
from hitlo.cost import compute_torque_curve

# EXPANDED ranges based on actual device capability
R_values = np.linspace(0.02, 0.40, 20)   # 2 cm to 40 cm
L0_values = np.linspace(0.18, 0.40, 20)  # 18 cm to 40 cm

peak_torque_grid = np.zeros((len(R_values), len(L0_values)))
k_rot_grid = np.zeros((len(R_values), len(L0_values)))

print(f"Sweeping {len(R_values)} R values x {len(L0_values)} L0 values = {len(R_values)*len(L0_values)} configs")
print(f"Spring stiffness: k = 10,500 N/m\n")

for i, R in enumerate(R_values):
    for j, L0 in enumerate(L0_values):
        angles, torques = compute_torque_curve(R, L0,
                                                angle_min=-25,
                                                angle_max=25,
                                                n_points=200)
        peak_idx = np.argmax(np.abs(torques))
        peak_torque = abs(torques[peak_idx])
        peak_angle_rad = abs(np.radians(angles[peak_idx]))
        if peak_angle_rad > 0.01:
            k_rot = peak_torque / peak_angle_rad
        else:
            k_rot = 0
        
        peak_torque_grid[i, j] = peak_torque
        k_rot_grid[i, j] = k_rot

print("=" * 60)
print("ROTATIONAL STIFFNESS RANGE")
print("=" * 60)
print(f"Min k_rot:  {k_rot_grid.min():.1f} N·m/rad")
print(f"Max k_rot:  {k_rot_grid.max():.1f} N·m/rad")
print(f"Mean k_rot: {k_rot_grid.mean():.1f} N·m/rad")
print()
print("PEAK TORQUE RANGE")
print("=" * 60)
print(f"Min peak torque:  {peak_torque_grid.min():.2f} N·m")
print(f"Max peak torque:  {peak_torque_grid.max():.2f} N·m")
print()

# Find configurations close to specific stiffness targets
targets = [50, 100, 150, 180, 210, 240]
print("CONFIGURATIONS NEAR TARGET STIFFNESSES")
print("=" * 60)
for target in targets:
    diff = np.abs(k_rot_grid - target)
    idx = np.unravel_index(diff.argmin(), diff.shape)
    R_match = R_values[idx[0]]
    L0_match = L0_values[idx[1]]
    k_match = k_rot_grid[idx]
    tau_match = peak_torque_grid[idx]
    if abs(k_match - target) < 30:
        print(f"  Target {target:>4} N·m/rad: R={R_match*100:.1f}cm, L0={L0_match*100:.1f}cm "
              f"-> k={k_match:.1f} N·m/rad, peak tau={tau_match:.2f} N·m")
    else:
        print(f"  Target {target:>4} N·m/rad: closest is R={R_match*100:.1f}cm, L0={L0_match*100:.1f}cm "
              f"-> k={k_match:.1f} N·m/rad (diff {abs(k_match-target):.0f})")

# Save heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im0 = axes[0].imshow(k_rot_grid, aspect='auto', origin='lower',
                     extent=[L0_values[0]*100, L0_values[-1]*100,
                             R_values[0]*100, R_values[-1]*100],
                     cmap='viridis')
axes[0].set_xlabel('L0 (cm)')
axes[0].set_ylabel('R (cm)')
axes[0].set_title('Rotational Stiffness (N·m/rad)')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(peak_torque_grid, aspect='auto', origin='lower',
                     extent=[L0_values[0]*100, L0_values[-1]*100,
                             R_values[0]*100, R_values[-1]*100],
                     cmap='plasma')
axes[1].set_xlabel('L0 (cm)')
axes[1].set_ylabel('R (cm)')
axes[1].set_title('Peak Torque (N·m)')
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig('/Users/maccamardo/HITLO_Symmetry/stiffness_map.png', dpi=150)
print(f"\n✅ Saved heatmap to /Users/maccamardo/HITLO_Symmetry/stiffness_map.png")








import sys
sys.path.insert(0, '/Users/maccamardo/HITLO_Symmetry')

import numpy as np
import matplotlib.pyplot as plt
from hitlo.cost import compute_torque_curve

target_R = 0.26   # 26 cm
target_L0 = 0.284 # 28.4 cm

angles, torques = compute_torque_curve(target_R, target_L0,
                                        angle_min=-20,
                                        angle_max=20,
                                        n_points=200)

# Plot — note your sign convention: negative angle = dorsiflexion
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(angles, torques, 'b-', linewidth=2.5, label='Assistive')
ax.plot(angles, -torques, 'r--', linewidth=2.5, label='EA (geometric inverse)')
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.axvspan(-10, 0, alpha=0.1, color='blue', label='Stance dorsiflexion (assistive engages)')
ax.axvspan(0, 15, alpha=0.1, color='red', label='Push-off plantarflexion (EA engages)')
ax.set_xlabel('Ankle Angle (deg)\n← Dorsiflexion        Plantarflexion →', fontsize=11)
ax.set_ylabel('Torque (N·m)', fontsize=11)
ax.set_title(f'Standardized LegExoNET Torque Profile\n'
             f'R = 26 cm, L₀ = 28 cm, k_rot ≈ 180 N·m/rad (Collins/Sawicki optimum)',
             fontsize=12)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/maccamardo/HITLO_Symmetry/standardized_180.png', dpi=150)
plt.show()
print("Saved torque profile plot")


import sys
sys.path.insert(0, '/Users/maccamardo/HITLO_Symmetry')

import numpy as np
from hitlo.cost import compute_exo_torque, compute_torque_curve

R_values = np.linspace(0.05, 0.40, 20)

# For each R, find the L0 that gives slack at neutral (torque ≈ 0 at 0°)
print("Finding configurations with slack spring at neutral (0°)")
print("=" * 70)

L0_test = np.linspace(0.10, 0.50, 200)

for R in R_values:
    # Find L0 where torque at 0° crosses zero
    torques_at_neutral = np.array([compute_exo_torque(0.0, R, L0) for L0 in L0_test])
    # Find smallest L0 where torque becomes 0
    zero_crossings = np.where(torques_at_neutral <= 0.01)[0]
    if len(zero_crossings) > 0:
        L0_slack = L0_test[zero_crossings[0]]
        # Compute peak torque at -10° (typical dorsiflexion peak)
        tau_peak_walking = abs(compute_exo_torque(-10.0, R, L0_slack))
        tau_peak_max = abs(compute_exo_torque(-20.0, R, L0_slack))
        # Effective stiffness at walking peak
        k_walking = tau_peak_walking / np.radians(10) if tau_peak_walking > 0 else 0
        print(f"R={R*100:.1f}cm: L0_slack={L0_slack*100:.2f}cm, "
              f"peak τ at -10°={tau_peak_walking:.1f} N·m, "
              f"k_rot at walking={k_walking:.1f} N·m/rad")
        







        
import numpy as np
import matplotlib.pyplot as plt

def compute_exo_torque_custom(ankle_angle_deg, R, L0, theta, attachment_ratio, k=10500.0):
    segment_length = 0.335
    ankle_x, ankle_y = 0.0, 0.0
    
    angle_rad = np.radians(-ankle_angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    R_ankle = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    heel_rel = np.array([-0.08, -0.05])
    toe_rel = np.array([segment_length - 0.08, -0.05])
    
    rotated_heel = R_ankle @ heel_rel + np.array([ankle_x, ankle_y])
    rotated_toe = R_ankle @ toe_rel + np.array([ankle_x, ankle_y])
    
    attach_x = rotated_heel[0] + attachment_ratio * (rotated_toe[0] - rotated_heel[0])
    attach_y = rotated_heel[1] + attachment_ratio * (rotated_toe[1] - rotated_heel[1])
    
    anchor_angle = theta - 90.0
    anchor_x = ankle_x + R * np.cos(np.radians(anchor_angle))
    anchor_y = ankle_y + R * np.sin(np.radians(anchor_angle))
    
    Ldist = np.sqrt((attach_x - anchor_x)**2 + (attach_y - anchor_y)**2)
    if Ldist <= 1e-6:
        return 0.0
    
    tension = k * max(Ldist - L0, 0.0)
    force_x = tension * (anchor_x - attach_x) / Ldist
    force_y = tension * (anchor_y - attach_y) / Ldist
    
    lever_x = attach_x - ankle_x
    lever_y = attach_y - ankle_y
    taudes = -(lever_x * force_y - lever_y * force_x)
    return float(taudes)


# Standardized assistive (reference)
R_a, L0_a, theta_a, attach_a = 0.38, 0.42, 196.0, -0.2

# Reference torque magnitudes
tau_target_pf = abs(compute_exo_torque_custom(-10.0, R_a, L0_a, theta_a, attach_a))
print(f"Reference: assistive peak at -10° DF = {tau_target_pf:.2f} N·m")
print(f"Searching for EA: dorsiflexor torque at +10° plantarflexion, ~{tau_target_pf:.0f} N·m magnitude")
print()

# Wide search across all four parameters
attach_options = np.linspace(0.0, 1.0, 11)
theta_options = np.linspace(0, 360, 73)  # 5° increments
R_values = np.linspace(0.05, 0.40, 15)
L0_values = np.linspace(0.10, 0.50, 17)

candidates = []

for attach_ea in attach_options:
    for theta_ea in theta_options:
        for R in R_values:
            for L0 in L0_values:
                tau_neutral = compute_exo_torque_custom(0.0, R, L0, theta_ea, attach_ea)
                tau_df = compute_exo_torque_custom(-10.0, R, L0, theta_ea, attach_ea)
                tau_pf = compute_exo_torque_custom(10.0, R, L0, theta_ea, attach_ea)
                
                # EA criteria:
                # 1. Slack-ish at neutral
                # 2. Strong dorsiflexor torque (negative torque) at PF
                # 3. Magnitude near assistive target
                if abs(tau_neutral) < 1.0 and tau_pf < -10.0 and abs(tau_df) < 5.0:
                    magnitude_match = abs(abs(tau_pf) - tau_target_pf)
                    candidates.append((magnitude_match, R, L0, theta_ea, attach_ea, tau_pf, tau_neutral, tau_df))

print(f"Found {len(candidates)} EA candidates")

if candidates:
    candidates.sort()  # sort by best magnitude match
    print()
    print("Top 15 candidates (best magnitude match to assistive):")
    print("=" * 100)
    print(f"{'#':<3}{'τ@-10°':<10}{'τ@0°':<10}{'τ@+10°':<10}{'R(cm)':<8}{'L0(cm)':<8}{'θ°':<8}{'attach':<8}{'magdiff':<10}")
    print("-" * 100)
    for i, (mdiff, R, L0, theta, attach, tau_pf, tau_n, tau_df) in enumerate(candidates[:15]):
        print(f"{i+1:<3}{tau_df:<10.2f}{tau_n:<10.2f}{tau_pf:<10.2f}"
              f"{R*100:<8.1f}{L0*100:<8.1f}{theta:<8.0f}{attach:<8.2f}{mdiff:<10.2f}")
    
    # Plot the best
    mdiff, R_ea, L0_ea, theta_ea, attach_ea, tau_pf, _, _ = candidates[0]
    
    angles = np.linspace(-15, 15, 200)
    torques_assistive = np.array([compute_exo_torque_custom(a, R_a, L0_a, theta_a, attach_a) 
                                  for a in angles])
    torques_ea = np.array([compute_exo_torque_custom(a, R_ea, L0_ea, theta_ea, attach_ea) 
                           for a in angles])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(angles, torques_assistive, 'b-', linewidth=2.5, label='Assistive')
    ax.plot(angles, torques_ea, 'r-', linewidth=2.5, label='EA (best match)')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Ankle Angle (deg)\n← Dorsiflexion          Plantarflexion →')
    ax.set_ylabel('Torque (N·m)')
    ax.set_title('Standardized LegExoNET Configurations: Assistive vs Inverted EA Profile\n'
                 f'Assistive: R={R_a*100:.0f}cm, L₀={L0_a*100:.0f}cm, θ={theta_a:.0f}°, attach={attach_a}\n'
                 f'EA: R={R_ea*100:.0f}cm, L₀={L0_ea*100:.1f}cm, θ={theta_ea:.0f}°, attach={attach_ea:.2f}')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/maccamardo/HITLO_Symmetry/inverted_torque_profile.png', dpi=150)
    plt.show()
else:
    print("Still no candidates — there may be a fundamental hardware limitation")