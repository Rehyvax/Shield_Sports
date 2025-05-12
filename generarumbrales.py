import numpy as np
import pandas as pd
import itertools
import json
from tqdm import tqdm

def get_variance_modulation(parameter, phase, level, weight, sex):
    # BASE FACTORS POR PARÁMETRO
    param_base = {
        'HR': 1.0,
        'HRV': 1.2,
        'Temperature': 0.8,
        'Sweat': 1.0,
        'Electrolytes': 1.0,
        'Ventilation': 1.0,
        'Cadence': 0.9,
        'Impact': 1.0,
        'KneeAcc': 1.0,
        'AnkleAcc': 1.0
    }.get(parameter, 1.0)

    # FACTOR POR FASE
    phase_factor = {
        0: 0.4,  # warm-up
        1: 0.8,
        2: 1.0,
        3: 1.5,  # peak
        4: 0.7,
        5: 0.9
    }.get(phase, 1.0)

    # FACTOR POR NIVEL
    level_factor = {
        'beginner': 1.6,
        'intermediate': 1.1,
        'advanced': 0.7
    }.get(level, 1.0)

    # FACTOR POR SEXO
    sex_factor = 1.05 if sex == 'male' else 1.0

    # FACTOR POR PESO
    weight_factor = 1.2 if weight > 85 else 0.8 if weight < 60 else 1.0

    # RESULTADO FINAL
    return param_base * phase_factor * level_factor * sex_factor * weight_factor
# 🧪 Simulación temporal (reemplaza con tu simulate_training real)
def simulate_training(user, duration_min=10):
    def apply_realistic_outlier(value, base_prob, phase, level, sex, weight, scale=1.2, noise_sd=0.1):
        phase_factor = 1.3 if phase == 3 else 1.15 if phase in [1, 2] else 0.9
        level_factor = {'beginner': 2.0, 'intermediate': 1.0, 'advanced': 0.5}[level]
        sex_factor = 1.1 if sex == 'female' else 1.0
        weight_factor = 1.2 if weight > 85 else 0.9 if weight < 60 else 1.0

        prob_outlier = base_prob * phase_factor * level_factor * sex_factor * weight_factor

        if phase == 0:
            return value  # no aplicamos outlier durante el warm-up
        if np.random.rand() < prob_outlier:
            return value * scale + np.random.normal(0, noise_sd)
        return value

    t = np.arange(0, duration_min * 60, 1)  # segundos
    age = user['age']
    sex = user['gender']
    weight = user['weight']
    height = user.get('height', 170)  # puedes añadirlo en el perfil
    level = user['level']

    fc_max = 220 - age - (5 if sex == 'female' else 0)
    bmi = weight / ((height / 100) ** 2)

    def hr_expected_rest():
        base = 72 if sex == 'female' else 68
        correction = {'beginner': +5, 'intermediate': 0, 'advanced': -5}[level]
        age_corr = 0.15 * (age - 30)
        return max(50, base + correction + age_corr)

    def hrv_expected_rest():
        base = 55 if sex == 'male' else 50
        correction = {'beginner': -15, 'intermediate': 0, 'advanced': +10}[level]
        age_corr = -0.6 * (age - 30)
        return max(20, base + correction + age_corr)

    hr_rest = hr_expected_rest()
    hrv_rest = hrv_expected_rest()

    
    fat = 0.1

    fatigue_rate = {'beginner': 0.002, 'intermediate': 0.0015, 'advanced': 0.001}[level]
    noise = np.random.normal

    borders = np.cumsum([0, 0.10, 0.40, 0.20, 0.10, 0.20]) * len(t)
    phases = np.zeros_like(t)
    for i, (start, end) in enumerate(zip(borders[:-1], borders[1:])):
        phases[(t >= start) & (t < end)] = i

    hr = []; hrv = []; fatigue = []
    temp = []; sweat_loss = []; electrolyte_loss_list = []; ventilation = []
    cadence = []; impact = []
    knee_acc_left = []; knee_acc_right = []
    ankle_acc_left = []; ankle_acc_right = []

    for i, second in enumerate(t):
        phase = phases[i]

        # Valores target según fase
        if phase == 0:  # warm-up
            hr_target = hr_rest + 0.25 * (fc_max - hr_rest)
            hrv_target = hrv_rest
            fat += fatigue_rate * 0.3
            var = 3
        elif phase in [1, 2]:  # steady
            hr_target = hr_rest + 0.5 * (fc_max - hr_rest)
            hrv_target = hrv_rest - 5
            fat += fatigue_rate * 0.6
            var = 4
        elif phase == 3:  # peak
            hr_target = hr_rest + 0.85 * (fc_max - hr_rest)
            hrv_target = hrv_rest - 15
            fat += fatigue_rate * 1.0
            var = 7
        elif phase == 4:  # recovery
            hr_target = hr_rest + 0.35 * (fc_max - hr_rest)
            hrv_target = hrv_rest + 10
            fat -= fatigue_rate * 0.5
            var = 4
        else:  # stable end
            hr_target = hr_rest + 0.5 * (fc_max - hr_rest)
            hrv_target = hrv_rest
            fat += fatigue_rate * 0.4
            var = 4

        fat = np.clip(fat, 0, 1)

        # HR & HRV
        hr_val = apply_realistic_outlier(
    noise(loc=hr_target, scale=var),
    base_prob=get_variance_modulation('HR', phase, level, weight, sex) * 0.0015,
    phase=phase,
    level=level,
    sex=sex,
    weight=weight,
    scale=1.15,
    noise_sd=2
)

        hrv_val = apply_realistic_outlier(
    noise(loc=hrv_target - fat*10, scale=var + 2),
    base_prob=get_variance_modulation('HRV', phase, level, weight, sex) * 0.0025,
    phase=phase,
    level=level,
    sex=sex,
    weight=weight,
    scale=0.75,
    noise_sd=2
)



        # Temperatura
        temp_rise = 0.01 + (0.005 * (1 if phase in [2, 3] else 0))
        temp_base = 36.5 + (fat * 1.5)
        temp_val = apply_realistic_outlier(
    temp_base + temp_rise * (i / 60) + noise(0, 0.05),
    base_prob=get_variance_modulation('Temperature', phase, level, weight, sex) * 0.0015,
    phase=phase,
    level=level,
    sex=sex,
    weight=weight,
    scale=1.02,
    noise_sd=0.05
)

       

        # Superficie corporal (m²)
        bsa = 0.007184 * (height ** 0.725) * (weight ** 0.425)

        # Sudoración y electrolitos
        sweat_base = {
    0: 1.5,
    1: 3,
    2: 6,
    3: 9,
    4: 2.5,
    5: 3.5
}[phase] * bsa * (1.0 if sex == 'male' else 0.85)

        sweat_rate = apply_realistic_outlier(
    sweat_base,
    base_prob=get_variance_modulation('Sweat', phase, level, weight, sex) * 0.002,
    phase=phase,
    level=level,
    sex=sex,
    weight=weight,
    scale=1.3,
    noise_sd=0.05
)
        electrolyte_loss = apply_realistic_outlier(
    sweat_rate * 50 / 1000,
    base_prob=get_variance_modulation('Electrolytes', phase, level, weight, sex) * 0.002,
    phase=phase,
    level=level,
    sex=sex,
    weight=weight,
    scale=1.25,
    noise_sd=0.01
)


        # Ventilación
        
        ve_val = apply_realistic_outlier(
        (6 if sex == 'female' else 8) + (fat * 12) + noise(0, 1.5),
        base_prob=get_variance_modulation('Ventilation', phase, level, weight, sex) * 0.002,
        phase=phase,
        level=level,
        sex=sex,
        weight=weight,
        scale=1.3,
        noise_sd=1.2
        )
        # Cadencia
        cadence_base = {
        'beginner': 160,
        'intermediate': 170,
        'advanced': 178
        }[level]

        fatigue_penalty = fat * 12
        phase_modifier = -3 if phase == 4 else (0 if phase in [1, 2] else -1)
        cadence_val = apply_realistic_outlier(
    cadence_base + phase_modifier - fatigue_penalty + noise(0, 1.8),
    base_prob=get_variance_modulation('Cadence', phase, level, weight, sex) * 0.0015,
    phase=phase,
    level=level,
    sex=sex,
    weight=weight,
    scale=0.92,
    noise_sd=1.5
)
        



        

        # Impacto (g)
        impact_base = {
        'beginner': 1.6,
        'intermediate': 2.0,
        'advanced': 2.4
        }[level]

        # Aumenta con fatiga, fase intensa y peso
        impact_val = apply_realistic_outlier(
    impact_base + 0.003 * i + 0.002 * weight + fat * 0.4 + (0.15 if phase == 3 else 0) + noise(0, 0.12),
    base_prob=get_variance_modulation('Impact', phase, level, weight, sex) * 0.0018,
    phase=phase,
    level=level,
    sex=sex,
    weight=weight,
    scale=1.25,
    noise_sd=0.1
)
        
        knee_left = apply_realistic_outlier(
                1.5 + 0.0005 * i + fat * 0.3 + noise(0, 0.15),
                base_prob=get_variance_modulation('KneeAcc', phase, level, weight, sex) * 0.0012,
                phase=phase,
                level=level,
                sex=sex,
                weight=weight,
                scale=1.25,
                noise_sd=0.15
            )
        knee_right = apply_realistic_outlier(
                1.5 + 0.0005 * i + fat * 0.3 + noise(0, 0.15),
                base_prob=get_variance_modulation('KneeAcc', phase, level, weight, sex) * 0.0012,
                phase=phase,
                level=level,
                sex=sex,
                weight=weight,
                scale=1.25,
                noise_sd=0.15
            )
        knee_left = np.nan
        knee_right = np.nan

        
        ankle_left = apply_realistic_outlier(
                1.8 + 0.0008 * i + fat * 0.35 + noise(0, 0.18),
                base_prob=get_variance_modulation('AnkleAcc', phase, level, weight, sex) * 0.0012,
                phase=phase,
                level=level,
                sex=sex,
                weight=weight,
                scale=1.25,
                noise_sd=0.15
            )
        ankle_right = apply_realistic_outlier(
                1.8 + 0.0008 * i + fat * 0.35 + noise(0, 0.18),
                base_prob=get_variance_modulation('AnkleAcc', phase, level, weight, sex) * 0.0012,
                phase=phase,
                level=level,
                sex=sex,
                weight=weight,
                scale=1.25,
                noise_sd=0.15
                )
        
        ankle_left = np.nan
        ankle_right = np.nan

        # Guardar en listas
        temp.append(temp_val)
        sweat_loss.append(sweat_rate)
        electrolyte_loss_list.append(electrolyte_loss)
        ventilation.append(ve_val)
        cadence.append(cadence_val)
        impact.append(impact_val)
        knee_acc_left.append(knee_left)
        knee_acc_right.append(knee_right)
        ankle_acc_left.append(ankle_left)
        ankle_acc_right.append(ankle_right)
        hr.append(np.clip(hr_val, 40, fc_max + 10))
        hrv.append(np.clip(hrv_val, 10, 120))
        fatigue.append(fat)

 

    return pd.DataFrame({
        'Time_s': t,
        'Heart_Rate_bpm': hr,
        'HRV_RMSSD_ms': hrv,
        'Estimated_Fatigue': fatigue,
        'Temperature_C': temp,
        'Sweat_Loss_ml_min': sweat_loss,
        'Electrolyte_Loss_mmol_L': electrolyte_loss_list,
        'Ventilation_L_min': ventilation,
        'Cadence_steps_min': cadence,
        'Impact_g': impact,
        'Knee_Acceleration_Left_g': knee_acc_left,
        'Knee_Acceleration_Right_g': knee_acc_right,
        'Ankle_Acceleration_Left_g': ankle_acc_left,
        'Ankle_Acceleration_Right_g': ankle_acc_right,
        'Phase': phases
        })


# 👤 Combinaciones
sexes = ["male", "female"]
levels = ["beginner", "intermediate", "advanced"]
ages = list(range(20, 65, 5))
weights = list(range(50, 101, 5))
heights = list(range(150, 191, 5))
durations = list(range(5, 31, 5))

combinations = list(itertools.product(sexes, levels, ages, weights, heights, durations))

parameters = [
    "HRV_RMSSD_ms",
    "Sweat_Loss_ml_min",
    "Electrolyte_Loss_mmol_L",
    "Cadence_steps_min",
    "Impact_g",
]

collected_data = {param: [] for param in parameters}

print(f"Ejecutando {len(combinations)} simulaciones...\n")

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
        collected_data[param].extend(df[param].values)

# 📊 Calcular percentil 95
adaptive_thresholds = {
    param: round(float(np.percentile(values, 95)), 3)
    for param, values in collected_data.items()
}

# 💾 Guardar en JSON
with open("adaptive_thresholds.json", "w") as f:
    json.dump(adaptive_thresholds, f, indent=4)

print("\n✅ Umbrales generados y guardados en 'adaptive_thresholds.json':")
for k, v in adaptive_thresholds.items():
    print(f"  - {k}: {v}")
