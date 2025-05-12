import numpy as np
import pandas as pd
import itertools
import json
from tqdm import tqdm

# -------------------
# FUNCIONES DE SIMULACIÓN
# -------------------

def get_variance_modulation(parameter, phase, level, weight, sex):
    param_base = {
        'HR': 1.0,
        'HRV': 1.2,
        'Ventilation': 1.0,
        'Impact': 1.0,
        'KneeAcc': 1.0,
        'AnkleAcc': 1.0
    }.get(parameter, 1.0)

    phase_factor = {0: 0.4, 1: 0.8, 2: 1.0, 3: 1.5, 4: 0.7, 5: 0.9}.get(phase, 1.0)
    level_factor = {'beginner': 1.6, 'intermediate': 1.1, 'advanced': 0.7}.get(level, 1.0)
    sex_factor = 1.05 if sex == 'male' else 1.0
    weight_factor = 1.2 if weight > 85 else 0.8 if weight < 60 else 1.0

    return param_base * phase_factor * level_factor * sex_factor * weight_factor

def simulate_training(user, duration_min=10):
    def apply_realistic_outlier(value, base_prob, phase, level, sex, weight, scale=1.2, noise_sd=0.1):
        phase_factor = 1.3 if phase == 3 else 1.15 if phase in [1, 2] else 0.9
        level_factor = {'beginner': 2.0, 'intermediate': 1.0, 'advanced': 0.5}[level]
        sex_factor = 1.1 if sex == 'female' else 1.0
        weight_factor = 1.2 if weight > 85 else 0.9 if weight < 60 else 1.0

        prob_outlier = base_prob * phase_factor * level_factor * sex_factor * weight_factor

        if phase == 0:
            return value
        if np.random.rand() < prob_outlier:
            return value * scale + np.random.normal(0, noise_sd)
        return value

    t = np.arange(0, duration_min * 60, 1)
    age = user['age']
    sex = user['gender']
    weight = user['weight']
    height = user.get('height', 170)
    level = user['level']

    fc_max = 220 - age - (5 if sex == 'female' else 0)

    hr = []; fatigue = []; ventilation = []
    knee_acc_left = []; knee_acc_right = []
    ankle_acc_left = []; ankle_acc_right = []

    fat = 0.1
    fatigue_rate = {'beginner': 0.002, 'intermediate': 0.0015, 'advanced': 0.001}[level]
    noise = np.random.normal

    borders = np.cumsum([0, 0.10, 0.40, 0.20, 0.10, 0.20]) * len(t)
    phases = np.zeros_like(t)
    for i, (start, end) in enumerate(zip(borders[:-1], borders[1:])):
        phases[(t >= start) & (t < end)] = i

    for i, second in enumerate(t):
        phase = phases[i]

        if phase == 0:
            fat += fatigue_rate * 0.3
            var = 3
        elif phase in [1, 2]:
            fat += fatigue_rate * 0.6
            var = 4
        elif phase == 3:
            fat += fatigue_rate * 1.0
            var = 7
        elif phase == 4:
            fat -= fatigue_rate * 0.5
            var = 4
        else:
            fat += fatigue_rate * 0.4
            var = 4

        fat = np.clip(fat, 0, 1)

        hr_target = 70 + fat * 100
        hr_val = apply_realistic_outlier(
            noise(loc=hr_target, scale=var),
            base_prob=get_variance_modulation('HR', phase, level, weight, sex) * 0.0015,
            phase=phase, level=level, sex=sex, weight=weight,
            scale=1.15, noise_sd=2
        )

        ve_val = apply_realistic_outlier(
            (6 if sex == 'female' else 8) + (fat * 12) + noise(0, 1.5),
            base_prob=get_variance_modulation('Ventilation', phase, level, weight, sex) * 0.002,
            phase=phase, level=level, sex=sex, weight=weight,
            scale=1.3, noise_sd=1.2
        )

        knee_left = apply_realistic_outlier(
            1.5 + 0.0005 * i + fat * 0.3 + noise(0, 0.15),
            base_prob=get_variance_modulation('KneeAcc', phase, level, weight, sex) * 0.0012,
            phase=phase, level=level, sex=sex, weight=weight,
            scale=1.25, noise_sd=0.15
        )
        knee_right = knee_left

        ankle_left = apply_realistic_outlier(
            1.8 + 0.0008 * i + fat * 0.35 + noise(0, 0.18),
            base_prob=get_variance_modulation('AnkleAcc', phase, level, weight, sex) * 0.0012,
            phase=phase, level=level, sex=sex, weight=weight,
            scale=1.25, noise_sd=0.15
        )
        ankle_right = ankle_left

        hr.append(np.clip(hr_val, 40, fc_max + 10))
        fatigue.append(fat)
        ventilation.append(ve_val)
        knee_acc_left.append(knee_left)
        knee_acc_right.append(knee_right)
        ankle_acc_left.append(ankle_left)
        ankle_acc_right.append(ankle_right)

    return pd.DataFrame({
        'Time_s': t,
        'Heart_Rate_bpm': hr,
        'Estimated_Fatigue': fatigue,
        'Ventilation_L_min': ventilation,
        'Knee_Acceleration_Left_g': knee_acc_left,
        'Knee_Acceleration_Right_g': knee_acc_right,
        'Ankle_Acceleration_Left_g': ankle_acc_left,
        'Ankle_Acceleration_Right_g': ankle_right,
        'Phase': phases
    })

# -------------------
# EJECUCIÓN DE SIMULACIONES
# -------------------

sexes = ["male", "female"]
levels = ["beginner", "intermediate", "advanced"]
ages = list(range(20, 65, 5))
weights = list(range(50, 101, 5))
heights = list(range(150, 191, 5))
durations = list(range(5, 31, 5))

combinations = list(itertools.product(sexes, levels, ages, weights, heights, durations))

parameters = [
    "Heart_Rate_bpm",
    "Estimated_Fatigue",
    "Ventilation_L_min",
    "Knee_Acceleration_Left_g",
    "Knee_Acceleration_Right_g",
    "Ankle_Acceleration_Left_g",
    "Ankle_Acceleration_Right_g"
]

collected_data = {param: [] for param in parameters}

print(f"🔁 Ejecutando {len(combinations)} simulaciones para parámetros extra...\n")

for sex, level, age, weight, height, duration in tqdm(combinations, desc="Simulando"):
    user = {
        "age": age,
        "gender": sex,
        "weight": weight,
        "height": height,
        "level": level,
    }

    df = simulate_training(user, duration)
    for param in parameters:
        values = df[param].dropna().values
        collected_data[param].extend(values)

adaptive_thresholds_extra = {
    param: round(float(np.percentile(values, 95)), 3)
    for param, values in collected_data.items()
}

with open("adaptive_thresholds_extra.json", "w") as f:
    json.dump(adaptive_thresholds_extra, f, indent=4)

print("\n✅ Umbrales extra generados y guardados en 'adaptive_thresholds_extra.json':")
for k, v in adaptive_thresholds_extra.items():
    print(f"  - {k}: {v}")
