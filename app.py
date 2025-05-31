import streamlit as st
import os
import json
import time
import base64
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
from datetime import datetime

USER_FILE = "users.json"

def load_users():
    if not os.path.exists(USER_FILE):
        with open(USER_FILE, "w") as f:
            json.dump({
                "admin": {
                    "password": "1234",
                    "training_frequency": 3,
                    "injuries": {"knee": False, "ankle": False, "tibia": False, "back": False}
                }
            }, f)
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)

def register_user(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = {
        "password": password,
        "training_frequency": 3,
        "injuries": {"knee": False, "ankle": False, "tibia": False, "back": False}
    }
    save_users(users)
    return True

# Inicialización del estado de sesión
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_type" not in st.session_state:
    st.session_state.user_type = None
if "start" not in st.session_state:
    st.session_state.start = False
if "show_summary" not in st.session_state:
    st.session_state.show_summary = False
if "block_training_loop" not in st.session_state:
    st.session_state.block_training_loop = False
if "view" not in st.session_state:
    st.session_state.view = "main"

# Inicializa campos sin sobreescribirlos
for field in ["New username", "New password", "Confirm password"]:
    if field not in st.session_state:
        st.session_state[field] = ""
if "show_terms" not in st.session_state:
    st.session_state["show_terms"] = False

# VISTA DE TÉRMINOS
if st.session_state.view == "terms":
    st.markdown("### 📄 Terms and Conditions")

    pdf_url = "https://raw.githubusercontent.com/Rehyvax/Smart_trainer/main/terms.pdf"
    st.markdown(
        f'<a href="{pdf_url}" target="_blank" rel="noopener noreferrer" style="font-size:16px;">'
        'View Terms and Conditions (PDF)</a>',
        unsafe_allow_html=True
    )

    if st.button("🔙 Back"):
        st.session_state.view = "main"
        st.session_state["show_terms"] = True
        st.rerun()

    st.stop()


# INTERFAZ DE LOGIN Y REGISTRO
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo.png", width=180)
        st.markdown("<h2 style='text-align: center; color: #FF6B00;'>Smart Trainer</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Log in to access the Smart Trainer platform</p>", unsafe_allow_html=True)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("🔐 Login"):
            users = load_users()
            if username in users and users[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.user_type = "admin"
                st.session_state.username = username
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid username or password")

        with st.expander("➕ Create new account"):
            new_user = st.text_input("New username", value=st.session_state["New username"])
            st.session_state["New username"] = new_user

            new_pass = st.text_input("New password", type="password", value=st.session_state["New password"])
            st.session_state["New password"] = new_pass

            confirm_pass = st.text_input("Confirm password", type="password", value=st.session_state["Confirm password"])
            st.session_state["Confirm password"] = confirm_pass

            st.markdown("<label style='font-weight: 500;'>📄 Terms and Conditions</label>", unsafe_allow_html=True)

            # Botón para mostrar el PDF
            if st.button("View Terms and Conditions"):
                st.session_state["view"] = "terms"
                st.rerun()

            # Mostrar checkbox solo si se ha visualizado el PDF
            accept_terms = False
            if st.session_state["show_terms"]:
                accept_terms = st.checkbox("I accept the Terms and Conditions")

            # Registro
            if st.button("Register"):
                if not new_user or not new_pass:
                    st.warning("Please fill all fields.")
                elif new_user in load_users():
                    st.warning("Username already exists.")
                elif new_pass != confirm_pass:
                    st.warning("Passwords do not match.")
                elif not st.session_state["show_terms"]:
                    st.warning("You must view the Terms and Conditions first.")
                elif not accept_terms:
                    st.warning("You must accept the Terms and Conditions.")
                else:
                    if register_user(new_user, new_pass):
                        st.success("✅ User registered successfully. You can now log in.")
                        time.sleep(1)
                        # Reiniciamos los estados
                        for key in ["New username", "New password", "Confirm password", "show_terms", "register_reset"]:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
                    else:
                        st.error("Error registering user.")

        st.stop()




view_mode = st.session_state.get("view", "main")
if view_mode == "summary" and st.session_state.get("finished_data") is None:
    st.markdown("⏳ Cargando resumen del entrenamiento...")
    st.stop()
st.markdown("""
<style>
/* Fondo global completamente negro */
html, body, .main, .block-container {
    background-color: #121212 !important;
    color: #FFFFFF !important;
}

/* Sidebar oscuro */
[data-testid="stSidebar"] {
    background-color: #1E1E1E !important;
    color: white !important;
    border-right: 1px solid #2E2E2E;
}

/* Botones naranja */
.stButton>button {
    background-color: #FF6B00 !important;
    color: white !important;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1rem;
}
.stButton>button:hover {
    background-color: #e65c00 !important;
}

/* Títulos en naranja */
h1, h2, h3, h4, h5 {
    color: #FF6B00 !important;
    font-weight: 700;
}

/* Etiquetas tipo 'Age', 'Gender'... */
label, .css-1v0mbdj, .css-1p4vugx, .css-1e5imcs {
    color: #FFFFFF !important;
    font-weight: 500;
}

/* Select boxes */
.css-1wa3eu0-placeholder {
    color: #FF6B00 !important;
}
.css-1dimb5e-singleValue {
    color: #FF6B00 !important;
}
.css-13cymwt-control {
    border: 1px solid #FF6B00 !important;
    background-color: #2A2A2A !important;
}

/* Sliders extremos e indicadores */
.css-1cpxqw2, .css-1n76uvr {
    color: #FF6B00 !important;
    font-weight: bold;
}

/* Etiquetas de checkboxes */
div[data-testid="stCheckbox"] label span {
    color: #FFFFFF !important;
    opacity: 1 !important;
    font-weight: 500;
}

/* Etiquetas de los radio buttons */
div[data-testid="stRadio"] label {
    color: #FFFFFF !important;
    opacity: 1 !important;
    font-weight: 500;
}

/* BLOQUES DE RESULTADOS / CONTENEDORES DINÁMICOS */
div[data-testid="stVerticalBlock"] > div[tabindex="0"] {
    background-color: #1E1E1E !important;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 0 10px rgba(0,0,0,0.2);
}

/* Texto dentro de las tarjetas de resultados */
div[data-testid="stVerticalBlock"] > div[tabindex="0"] * {
    color: #FF6B00 !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


users = load_users()
user = users.get(st.session_state.username, {})

with open("adaptive_thresholds_all.json") as f:
    adaptive_thresholds = json.load(f)

freq = user.get("training_frequency", 3)
injuries = user.get("injuries", {})

# Copiar umbrales para no modificar el original
adjusted_thresholds = adaptive_thresholds.copy()

# Ajuste por frecuencia
if freq >= 4:
    adjusted_thresholds["Impact_g"] = adaptive_thresholds["Impact_g"] * 0.92

# Ajuste por lesión general (cualquier lesión)
if any(injuries.values()):
    adjusted_thresholds["Estimated_Fatigue"] = adaptive_thresholds["Estimated_Fatigue"] * 0.92
    adjusted_thresholds["Impact_g"] = adjusted_thresholds["Impact_g"] * 0.92  # Reducimos aún más impacto si hay lesión

# Ajuste por lesión específica en rodilla o tobillo
if injuries.get("knee", False):
    adjusted_thresholds["Knee_Acceleration_Left_g"] = adaptive_thresholds.get("Knee_Acceleration_Left_g", 2.0) * 0.92
    adjusted_thresholds["Knee_Acceleration_Right_g"] = adaptive_thresholds.get("Knee_Acceleration_Right_g", 2.0) * 0.92

if injuries.get("ankle", False):
    adjusted_thresholds["Ankle_Acceleration_Left_g"] = adaptive_thresholds.get("Ankle_Acceleration_Left_g", 2.0) * 0.92
    adjusted_thresholds["Ankle_Acceleration_Right_g"] = adaptive_thresholds.get("Ankle_Acceleration_Right_g", 2.0) * 0.92



# Botones de perfil y exit arriba a la derecha
if st.session_state.logged_in and not st.session_state.start and st.session_state.get("view") != "summary":
    button_container = st.empty()

    with button_container:
        top_col1, top_col2, top_col3 = st.columns([9, 1, 1])
        with top_col3:
            with st.expander(f"👤 {st.session_state.username}", expanded=False):
                st.markdown(f"**Username:** `{st.session_state.username}`")
                if st.button("🗑 Delete Account"):
                    users = load_users()
                    if st.session_state.username in users:
                        del users[st.session_state.username]
                        save_users(users)
                        st.success("User deleted. Restarting...")
                        time.sleep(1)
                        st.session_state.clear()
                        st.rerun()
                users = load_users()
            user_data = users.get(st.session_state.username, {})
            freq_train = user_data.get("training_frequency", 3)
            injuries = user_data.get("injuries", {"knee": False, "ankle": False, "tibia": False, "back": False})

            with st.expander("Additional information", expanded=False):
                freq_train_new = st.slider("Training frequency per week", 0, 14, freq_train)
                knee_new = st.checkbox("Knee injury", value=injuries.get("knee", False))
                ankle_new = st.checkbox("Ankle injury", value=injuries.get("ankle", False))
                tibia_new = st.checkbox("Tibia injury", value=injuries.get("tibia", False))
                back_new = st.checkbox("Back injury", value=injuries.get("back", False))

                # Guardar cambios si hay diferencias
                if (freq_train_new != freq_train or knee_new != injuries.get("knee", False) or
                    ankle_new != injuries.get("ankle", False) or tibia_new != injuries.get("tibia", False) or
                    back_new != injuries.get("back", False)):
                    
                    users[st.session_state.username]["training_frequency"] = freq_train_new
                    users[st.session_state.username]["injuries"] = {
                        "knee": knee_new,
                        "ankle": ankle_new,
                        "tibia": tibia_new,
                        "back": back_new
                    }
                    save_users(users)
                    st.success("User info updated!")

            if st.button("🚪 Exit"):
                st.session_state.clear()
                st.rerun()

        # Contenido central estilizado
        logo_col1, logo_col2, logo_col3 = st.columns([1, 2, 1])
        with logo_col2:
            st.image("logo.png", width=220)
            st.markdown("<h1 style='color: #FF6B00; text-align: center;'>Smart Trainer</h1>", unsafe_allow_html=True)
            st.markdown("<p style='color: gray; text-align: center;'>Injury prevention assistant</p>", unsafe_allow_html=True)




for key in ["start", "index", "data", "locked_user", "favorites"]:
    if key not in st.session_state:
        st.session_state[key] = set() if key == "favorites" else False if key == "start" else 0 if key == "index" else None
# ----------------------
# UTILITY FUNCTIONS
# ----------------------
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
    sex_factor = 1.05 if sex == 'male' else 1.1

    # FACTOR POR PESO
    weight_factor = 1.2 if weight > 85 else 0.8 if weight < 60 else 1.0

    # RESULTADO FINAL
    return param_base * phase_factor * level_factor * sex_factor * weight_factor

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
    dur_fase_warmup = int(0.1 * len(t))
    age = user['age']
    sex = user['gender']
    weight = user['weight']
    height = user.get('height', 170)  # puedes añadirlo en el perfil
    level = user['level']

    fc_max = 230 - age - (5 if sex == 'female' else 0)
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

        # Inicializar contador de segundos en recovery
        if i == 0 or phases[i - 1] != 4:
            time_in_recovery_hr = 0
        else:
            time_in_recovery_hr += 1

        # Valores target según fase
        if phase == 0:  # warm-up progresivo
            progress = i / dur_fase_warmup
            progress = np.clip(progress, 0, 1)
            hr_target = hr_rest + progress * (0.65 * (fc_max - hr_rest))
            hrv_target = hrv_rest
            fat += fatigue_rate * 0.3
            var = 3
        elif phase in [1, 2]:  # steady
            hr_target = hr_rest + 0.82 * (fc_max - hr_rest)
            hrv_target = hrv_rest - 5
            fat += fatigue_rate * 0.6
            var = 4
        elif phase == 3:  # peak
            hr_target = hr_rest + 0.95 * (fc_max - hr_rest)
            hrv_target = hrv_rest - 15
            fat += fatigue_rate * 1.0
            var = 7
        elif phase == 4:  # recovery
            # Bajada progresiva desde la frecuencia previa
            decay_rate = 0.1  # bpm por segundo (ajustable según realismo)
            target_min = hr_rest + 0.35 * (fc_max - hr_rest)
            hr_target = max(prev_hr_target - decay_rate * time_in_recovery_hr, target_min)
            hrv_target = hrv_rest + 10
            fat -= fatigue_rate * 0.5
            var = 4
        else:  # stable end
            hr_target = hr_rest + 0.5 * (fc_max - hr_rest)
            hrv_target = hrv_rest
            fat += fatigue_rate * 0.4
            var = 4


        # Limitar fatiga entre 0 y 1
        fat = np.clip(fat, 0, 1)

        # Guardar el valor anterior de HR para simular continuidad
        prev_hr_target = hr_target

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
        temp_rise = 0.008 + (0.003 * (1 if phase in [2, 3] else 0))
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
    0: 3.5,
    1: 5.5,
    2: 7.5,
    3: 9.5,
    4: 4.5,
    5: 4
}[phase] * bsa * (1.2 if sex == 'male' else 0.6)

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
        if st.session_state.get('chest_band'):
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
        else:
            ve_val = np.nan
        # Cadencia
        cadence_base = {
        'beginner': 160,
        'intermediate': 170,
        'advanced': 178
        }[level] - (6 if sex == 'female' else 0)

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

        # Contador de tiempo dentro de recovery
        if i == 0 or phases[i - 1] != 4:
            time_in_recovery = 0
        else:
            time_in_recovery += 1

        # Crecimiento de impacto solo fuera de recovery
        time_growth = 0.003 * i if phase != 4 else 0

        recovery_decrease = min(0.0075 * time_in_recovery, 2.5)  # puedes ajustar a 2.0 si aún baja demasiado

        # Cálculo previo del valor crudo de impacto
        raw_impact = (
            impact_base
            + time_growth
            + 0.002 * weight
            + fat * 0.4
            + (0.15 if phase == 3 else 0)
            - recovery_decrease
            + noise(0, 0.12)
        )

        # Limitar a un mínimo fisiológico razonable
        raw_impact = max(raw_impact, 0.5)

        # Aplicar variabilidad fisiológica final
        impact_val = apply_realistic_outlier(
            raw_impact,
            base_prob=get_variance_modulation('Impact', phase, level, weight, sex) * 0.0018,
            phase=phase,
            level=level,
            sex=sex,
            weight=weight,
            scale=1.25,
            noise_sd=0.1
        )
        if st.session_state.get('knee_band'):
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
        else:
            knee_left = np.nan
            knee_right = np.nan

        if st.session_state.get('ankle_band'):
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
        else:
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


# ----------------------
# RISK MONITOR PLACEHOLDER (Safe Init Without Premature Render)
# ----------------------

risk_box = st.empty()  # No rendering yet, only prepare placeholder
live_box = st.empty()  # También lo inicializamos aquí
# ----------------------
# PLACEHOLDER INIT TO FIX NameError
# ----------------------

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'risk_status' not in st.session_state:
    st.session_state.risk_status = "✅ Parameters stable"
if 'risk_box' not in st.session_state:
    st.session_state.risk_box = st.empty()

# Asignar variables
messages = st.session_state.messages
risk_status = st.session_state.risk_status
actions = actions if 'actions' in locals() else []

# Evitar duplicar acciones
if actions and (not hasattr(st.session_state, 'last_actions') or st.session_state.last_actions != actions):
    st.session_state.last_actions = actions
else:
    actions = []

# Crear el HTML de alerta
status_color = 'red' if messages else 'green'
risk_html = f"""
<div id='risk-box' style='padding:1rem;background-color:#1E1E1E;border-radius:10px; color:#FF6B00; font-weight:500; margin-bottom:8px; animation: {'pulse 1s infinite' if messages else 'none'};'>
    <b>Injury Risk Monitor:</b> <span style='color:{status_color}'><b>{risk_status}</b></span><br>
    {('<br>'.join(messages)) if messages else 'All values within safe range.'}
    {"<hr style='margin-top:8px'><b>Clinical Recommendations:</b>" if actions else ''}
    {"<ul style='padding-left:1.2rem; line-height:1.6; font-size:0.95rem;'>" + ''.join([f"<li>{a}</li>" for a in actions]) + "</ul>" if actions else ''}
</div>
"""


# Mostrar solo si se ha iniciado el entrenamiento
if st.session_state.start:
    st.session_state.risk_box.markdown(risk_html, unsafe_allow_html=True)

# ----------------------
# DISPLAY SUMMARY FUNCTION
# ----------------------


def display_summary(df):
    

    fig, ax = plt.subplots(figsize=(10, 4))
    phase_colors = ['#e8f5e9', '#fffde7', '#ffebee', '#e3f2fd', '#ede7f6']
    phase_names = ['Warm-up', 'Steady', 'Intense', 'Peak', 'Recovery']
    times = df['Time_s'].values
    phases = df['Phase'].values

    for p in range(5):
        mask = phases == p
        if np.any(mask):
            ax.axvspan(times[mask][0], times[mask][-1], color=phase_colors[p], alpha=0.45)

    for col in [
    "Heart_Rate_bpm",
    "HRV_RMSSD_ms",
    "Cadence_steps_min",
    "Estimated_Fatigue",
    "Impact_g",
    "Ventilation_L_min",
    "Sweat_Loss_ml_min",
    "Electrolyte_Loss_mmol_L"
]:
        if col in df.columns:
            ax.plot(df["Time_s"], df[col], label=col.replace('_', ' '), linewidth=1.5)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("Training Parameters Over Time", fontsize=13, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9)

    st.markdown("""
    <div style='display: flex; justify-content: center; gap: 25px; font-size: 14px; margin-top: 1rem; margin-bottom: 1rem;'>
        <div><span style='display:inline-block;width:16px;height:14px;background-color:#e8f5e9;margin-right:6px;border-radius:2px;'></span>Warm-up</div>
        <div><span style='display:inline-block;width:16px;height:14px;background-color:#fffde7;margin-right:6px;border-radius:2px;'></span>Steady</div>
        <div><span style='display:inline-block;width:16px;height:14px;background-color:#ffebee;margin-right:6px;border-radius:2px;'></span>Intense</div>
        <div><span style='display:inline-block;width:16px;height:14px;background-color:#e3f2fd;margin-right:6px;border-radius:2px;'></span>Peak</div>
        <div><span style='display:inline-block;width:16px;height:14px;background-color:#ede7f6;margin-right:6px;border-radius:2px;'></span>Recovery</div>
    </div>
    """, unsafe_allow_html=True)

    st.pyplot(fig)

    # Este bloque de variaciones de temperatura/fatiga también lo podemos quitar si lo moverás al nuevo dashboard


# ----------------------
# INJURY REPORT FUNCTION
# ----------------------

def generate_injury_report(df, user):
    
    age = user['age']
    sex = user['gender']
    weight = user['weight']
    height = user['height']
    level = user['level']
    fc_max = 220 - age - (5 if sex == 'female' else 0)

    alerts = []
    suggestions = []

    def check(param, value, threshold, direction, msg, adv):
        if (direction == '>' and value > threshold) or (direction == '<' and value < threshold):
            alerts.append(f"⚠️ {msg}: {value:.2f}")
            suggestions.append(adv)

    # Fase dominante para ajustar umbrales
    phase = int(df['Phase'].mode()[0]) if 'Phase' in df.columns else 1

    # Factores de ajuste de umbrales
    phase_factor = 1.3 if phase == 3 else 1.1 if phase in [1, 2] else 0.9
    level_factor = {'beginner': 1.2, 'intermediate': 1.0, 'advanced': 0.8}[level]
    sex_factor = 1.1 if sex == 'male' else 1.05
    weight_factor = 1.2 if weight > 85 else 0.9 if weight < 60 else 1.0

    # Eliminamos el factor de tiempo y usamos solo la fase y perfil para ajustar el umbral
    def adjust(base):
        return base * phase_factor * level_factor * sex_factor * weight_factor
    def adjust_temp(base):
        return base * level_factor * sex_factor * weight_factor  # ❌ sin phase_factor

    # MÉTRICAS CARDIO
    hr = df['Heart_Rate_bpm'].mean()
    hrv = df['HRV_RMSSD_ms'].mean()
    fatiga = df['Estimated_Fatigue'].mean()

    check("HR", hr, adjust(0.9 * fc_max), '>', "High Heart Rate", "Reduce pace and increase recovery intervals.")
    check("HRV", hrv, adjust(adaptive_thresholds["HRV_RMSSD_ms"]), '<', "Low HRV (stress/fatigue)", "Incorporate rest or breathing exercises.")
    check("Fatigue", fatiga, adjust(adaptive_thresholds["Estimated_Fatigue"]), '>', "High Fatigue Load", "Consider lowering training load.")

    # TÉRMICO / HIDRATACIÓN
    temp = df['Temperature_C'].mean()
    sweat = df['Sweat_Loss_ml_min'].mean()
    elec = df['Electrolyte_Loss_mmol_L'].mean()
    ve = df['Ventilation_L_min'].dropna().mean()

    check("Temp", temp, adjust_temp(adaptive_thresholds["Temperature_C"]), '>', "Core temperature too high", "Stop and cool down.")
    check("Sweat", sweat, adjust(adaptive_thresholds["Sweat_Loss_ml_min"]), '>', "Excessive sweating", "Hydrate regularly and use electrolyte drinks.")
    check("Electrolytes", elec, adjust(adaptive_thresholds["Electrolyte_Loss_mmol_L"]), '>', "High electrolyte loss", "Consider salt supplementation.")

    if not np.isnan(ve):
        check("Ventilation", ve, adjust(adaptive_thresholds["Ventilation_L_min"]), '>', "High ventilation rate", "Control breathing and posture.")

    # MOVIMIENTO
    cadence = df['Cadence_steps_min'].mean()
    impact = df['Impact_g'].mean()

    check("Cadence", cadence, adaptive_thresholds["Cadence_steps_min"], '<', "Cadence too low", "Focus on shorter strides and faster rhythm.")
    check("Impact", impact, adjust(adaptive_thresholds["Impact_g"]), '>', "Impact too high", "Use softer landing and improve technique.")

    # ASIMETRÍAS
    def check_asym(left, right, name):
        if not left.empty and not right.empty:
            diff = np.abs(left.mean() - right.mean())
            rel_diff = diff / max(left.mean(), right.mean())
            if rel_diff > 0.25:
                alerts.append(f"⚠️ {name} acceleration asymmetry ({rel_diff*100:.1f}%)")
                suggestions.append(f"Work on symmetry in {name.lower()} movement and stability.")

    check_asym(df['Knee_Acceleration_Left_g'].dropna(), df['Knee_Acceleration_Right_g'].dropna(), "Knee")
    check_asym(df['Ankle_Acceleration_Left_g'].dropna(), df['Ankle_Acceleration_Right_g'].dropna(), "Ankle")
    from collections import Counter

    # Unimos todas las alertas de la sesión
    all_alerts = alerts.copy()
    if "alert_history" in st.session_state:
        for hist_alerts, _ in st.session_state.alert_history:
            all_alerts.extend(hist_alerts)

    # Agrupamos y eliminamos repeticiones exactas
    unique_alerts = list(dict.fromkeys(all_alerts))  # mantiene orden y elimina duplicados
    alert_counter = Counter(all_alerts)

    if unique_alerts:
        st.error(" Risk Factors Detected")
        st.subheader(" Injury Alert Summary")

        for a in unique_alerts:
            st.markdown(f"{a}")

        # Mostramos la más frecuente
        most_common_alert, _ = alert_counter.most_common(1)[0]
        st.markdown(f"**Most frequent issue:** {most_common_alert}")
    else:
        st.success("✅ No significant risk detected. All parameters within expected ranges.")





# ----------------------
# SESSION STATE INIT
# ----------------------



# ----------------------
# EXTRA SENSORS
# ----------------------

if not st.session_state.start:
    st.sidebar.markdown("### Accessories")
    knee_band = st.sidebar.checkbox("Knee Band")
    knee_side = None
    if knee_band:
        knee_side = st.sidebar.radio("Knee Band Side", ["Left", "Right"], horizontal=True)
        st.session_state.knee_side = knee_side
    ankle_band = st.sidebar.checkbox("Ankle Band")
    ankle_side = None
    if ankle_band:
        ankle_side = st.sidebar.radio("Ankle Band Side", ["Left", "Right"], horizontal=True)
        st.session_state.ankle_side = ankle_side
    st.session_state.knee_band = knee_band
    st.session_state.ankle_band = ankle_band
    chest_band = st.sidebar.checkbox("Chest Band")
    st.session_state.chest_band = chest_band


# ----------------------

st.sidebar.title("User Profile")
if not st.session_state.start:
    age = st.sidebar.slider("Age", 16, 75, 30)
    gender = st.sidebar.selectbox("Gender", ["male", "female"])
    weight = st.sidebar.slider("Weight (kg)", 40, 120, 70)
    height = st.sidebar.slider("Height (cm)", 140, 210, 170)
    level = st.sidebar.selectbox("Level", ["beginner", "intermediate", "advanced"])
    user = {"age": age, "gender": gender, "weight": weight, "height": height,"level": level}
else:
    user = st.session_state.locked_user

st.sidebar.markdown("---")

# ----------------------
# START/STOP BUTTONS
# ----------------------

duration = st.sidebar.slider("Training Duration (min)", 1, 60, 10)

if st.sidebar.button("▶ Start Training"):
    st.session_state.duration = duration
    st.session_state.data = simulate_training(user, duration_min=st.session_state.duration)
    st.session_state.index = 0
    st.session_state.start = True
    st.session_state.locked_user = user
    st.session_state.alert_history = []
    st.session_state.alert_timestamps = []
    st.session_state.finished_data = None  # limpiar datos previos
    st.rerun()

if st.sidebar.button("⏹ Stop Training"):
    # Comprobar si el entrenamiento ha comenzado
    if not st.session_state.start:
        st.warning("You must start the training first!")
    else:
        # Comprobar si hay datos de entrenamiento disponibles
        if st.session_state.data is not None and not st.session_state.data.empty:
            st.session_state.start = False
            folder = "trainings"
            os.makedirs(folder, exist_ok=True)
            df_save = st.session_state.data.iloc[:st.session_state.index + 1].copy()
            if not df_save.empty:
                st.session_state.finished_data = df_save
                st.session_state.locked_user = user
                filename = datetime.now().strftime("%d-%m-%Y %H-%M") + ".csv"
                path = os.path.join(folder, filename)
                df_save.to_csv(path, index=False)
            st.session_state.show_summary = True
            st.session_state.block_training_loop = True
            st.session_state.view = "summary"
            st.rerun()
        else:
            st.warning("No training data available. Start training first!")
    
if st.session_state.get("view") == "summary":
    if st.session_state.get("finished_data") is None:
        st.markdown("⏳ Cargando resumen del entrenamiento...")
        st.stop()
    else:
        df_save = st.session_state.finished_data
        user = st.session_state.locked_user
        df_save = st.session_state.finished_data
        user = st.session_state.locked_user
        age = user['age']
        sex = user['gender']
        fc_max = 220 - age - (5 if sex == 'female' else 0)

        tab1, tab2 = st.tabs(["📈 Graph Evolution", "📊 Review Parameters"])

        with tab1:
            st.markdown("### 📈 Training Parameter Evolution")
            display_summary(df_save)
            st.markdown("###  Injury Risk Report")
            generate_injury_report(df_save, user)


            # Mostrar porcentaje estimado
            total_time = df_save["Time_s"].iloc[-1]
            alert_time = len(st.session_state.alert_timestamps) * (total_time / len(df_save))
            risk_pct = round((alert_time / total_time) * 100, 1)
            
        with tab2:
            st.markdown("### Post-Training Metrics")
            st.session_state.messages = []
            st.session_state.risk_status = "✅ Parameters stable"
            st.session_state.actions = []
            
            avg_temp = round(df_save["Temperature_C"].mean(), 2)
            st.markdown(f"**🌡️ Average Body Temperature:** {avg_temp} °C")

            if not st.session_state.start:
                st.session_state.risk_box.empty()

            def render_metric(title, desc, fig, key):
                with st.container():
                    st.markdown(f"""
                    <div style='min-height:90px; margin-bottom:-25px'>
                        <h5>{title}</h5>
                        <p style='color:gray; font-size:13px; line-height:1.2;'>{desc}</p>
                    </div>
                """, unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True, key=key)

            def gauge_figure(value, unit, axis_range, color, steps):
                is_nan = np.isnan(value)
                display_value = 0 if is_nan else value

                return go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=display_value,
                    number={
                        'font': {'size': 22, 'color': "#FF6B00"},
                        'suffix': unit if not is_nan else "",
                        'prefix': "–" if is_nan else "",
                        'valueformat': ".2f"
                    },
                    gauge={
                        'axis': {
                            'range': axis_range,
                            'tickcolor': '#CCCCCC',
                            'tickfont': {'color': 'white'}
                        },
                        'bar': {
                            'color': "#FF6B00",
                            'thickness': 0.25
                        },
                        'bgcolor': "#121212",
                        'steps': [{'range': s['range'], 'color': '#00CED1'} for s in steps],
                        'threshold': {
                            'line': {'color': "white", 'width': 2},
                            'thickness': 0.75,
                            'value': display_value
                        }
                    },
                    domain={'x': [0, 1], 'y': [0, 1]},
                )).update_layout(paper_bgcolor="#121212", font={'color': "#FFFFFF"})

            metrics = [
                        (" Heart Rate", "bpm", df_save["Heart_Rate_bpm"].mean(), [40, fc_max + 10], "red", [
                {'range': [40, 0.6 * fc_max], 'color': "#8BC34A"},
                {'range': [0.6 * fc_max, 0.8 * fc_max], 'color': "#FFC107"},
                {'range': [0.8 * fc_max, fc_max + 10], 'color': "#F44336"},
            ], "Heart rate relative to your estimated max. High values indicate overexertion."),

            (" HRV (RMSSD)", "ms", df_save["HRV_RMSSD_ms"].mean(), [10, 100], "blue", [
                {'range': [10, 30], 'color': "#F44336"},
                {'range': [30, 50], 'color': "#FFC107"},
                {'range': [50, 100], 'color': "#8BC34A"},
            ], "Higher variability reflects better recovery and lower stress."),

            (" Fatigue", "", df_save["Estimated_Fatigue"].mean(), [0, 1], "orange", [
                {'range': [0, 0.4], 'color': "#8BC34A"},
                {'range': [0.4, 0.7], 'color': "#FFC107"},
                {'range': [0.7, 1], 'color': "#F44336"},
            ], "Accumulated physiological load. Higher levels indicate need for recovery."),

            (" Sweat Loss", "ml/min", df_save["Sweat_Loss_ml_min"].mean(), [0, 10], "blue", [
                {'range': [0, 3], 'color': "#8BC34A"},
                {'range': [3, 6], 'color': "#FFC107"},
                {'range': [6, 10], 'color': "#F44336"},
            ], "Estimated fluid loss. High values require regular hydration."),

            (" Electrolyte Loss", "mmol/L", df_save["Electrolyte_Loss_mmol_L"].mean(), [0, 1], "blue", [
                {'range': [0, 0.3], 'color': "#8BC34A"},
                {'range': [0.3, 0.5], 'color': "#FFC107"},
                {'range': [0.5, 1], 'color': "#F44336"},
            ], "Electrolytes lost through sweat. Important to replenish when high."),

            (" Cadence", "spm", df_save["Cadence_steps_min"].mean(), [130, 190], "purple", [
                {'range': [130, 150], 'color': "#F44336"},
                {'range': [150, 165], 'color': "#FFC107"},
                {'range': [165, 190], 'color': "#8BC34A"},
            ], "Steps per minute. A stable cadence reduces injury risk."),

            (" Impact", "g", df_save["Impact_g"].mean(), [0, 4], "brown", [
                {'range': [0, 1.8], 'color': "#8BC34A"},
                {'range': [1.8, 2.5], 'color': "#FFC107"},
                {'range': [2.5, 4], 'color': "#F44336"},
            ], "Force per stride. Impact stresses joints."),

            (" Knee Left", "g", df_save["Knee_Acceleration_Left_g"].mean() if "Knee_Acceleration_Left_g" in df_save.columns else float("nan"), [0, 4], "brown", [

                {'range': [0, 1.8], 'color': "#8BC34A"},
                {'range': [1.8, 2.5], 'color': "#FFC107"},
                {'range': [2.5, 4], 'color': "#F44336"},
            ], "Acceleration of left knee during motion."),

            (" Knee Right", "g", df_save["Knee_Acceleration_Right_g"].mean() if "Knee_Acceleration_Right_g" in df_save.columns else float("nan"), [0, 4], "brown", [

                {'range': [0, 1.8], 'color': "#8BC34A"},
                {'range': [1.8, 2.5], 'color': "#FFC107"},
                {'range': [2.5, 4], 'color': "#F44336"},
            ], "Acceleration of right knee during motion."),

            (" Ankle Left", "g", df_save["Ankle_Acceleration_Left_g"].mean() if "Ankle_Acceleration_Left_g" in df_save.columns else float("nan"), [0, 4], "brown", [

                {'range': [0, 1.8], 'color': "#8BC34A"},
                {'range': [1.8, 2.5], 'color': "#FFC107"},
                {'range': [2.5, 4], 'color': "#F44336"},
            ], "Acceleration of the left ankle during motion."),

            (" Ankle Right", "g", df_save["Ankle_Acceleration_Right_g"].mean() if "Ankle_Acceleration_Right_g" in df_save.columns else float("nan"), [0, 4], "brown", [

                {'range': [0, 1.8], 'color': "#8BC34A"},
                {'range': [1.8, 2.5], 'color': "#FFC107"},
                {'range': [2.5, 4], 'color': "#F44336"},
            ], "Acceleration of right ankle during motion."),

            (" Ventilation", "L/min", df_save["Ventilation_L_min"].mean() if "Ventilation_L_min" in df_save.columns else float("nan"), [5, 25], "blue", [

                {'range': [5, 12], 'color': "#8BC34A"},
                {'range': [12, 18], 'color': "#FFC107"},
                {'range': [18, 25], 'color': "#F44336"},
            ], "Air volume inhaled and exhaled per minute."),
            ]

            for i in range(0, len(metrics), 3):  # de 3 en 3
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(metrics):
                        with cols[j]:
                            name, unit, value, rng, color, steps, desc = metrics[i + j]
                            fig = gauge_figure(value, unit, rng, color, steps)
                            render_metric(name, desc, fig, key=f"metric_{i+j}")
            st.session_state.show_summary = False  # 🔁 Reseteamos el flag
            st.session_state.block_training_loop = False
            st.session_state.view = "main"
            st.markdown("---")

            if st.button("🔁 Volver al inicio"):
                st.session_state.view = "main"
                st.session_state.finished_data = None
                st.session_state.locked_user = None
                st.session_state.data = None
                st.session_state.index = 0
                st.session_state.messages = []
                st.session_state.risk_status = "✅ Parameters stable"
                st.session_state.alert_history = []
                st.rerun()
    



            

# ----------------------
# INJURY RISK ASSESSMENT FUNCTION
# ----------------------

def evaluate_injury_risk(window_df, user):
    age = user['age']
    sex = user['gender']
    weight = user['weight']
    height = user['height']
    level = user['level']
    fc_max = 220 - age - (5 if sex == 'female' else 0)

    alerts = []
    actions = []

    def check(condition, message, advice):
        if condition:
            alerts.append(f"\u26a0\ufe0f {message}")
            actions.extend(advice)

    # Obtener fase dominante de la ventana actual
    phase = int(window_df['Phase'].mode()[0]) if 'Phase' in window_df.columns else 1

    # Factores de ajuste de umbrales
    phase_factor = 1.3 if phase == 3 else 1.1 if phase in [1, 2] else 0.9
    level_factor = {'beginner': 1.2, 'intermediate': 1.0, 'advanced': 0.8}[level]
    sex_factor = 1.1 if sex == 'male' else 0.95
    weight_factor = 1.2 if weight > 85 else 0.9 if weight < 60 else 1.0

    def adjust(base):
        return base * phase_factor * level_factor * sex_factor * weight_factor
    def adjust_temp(base):
        return base * level_factor * sex_factor * weight_factor  

    # --------- FACTORES CARDÍACOS ----------
    hr_mean = window_df['Heart_Rate_bpm'].mean()
    hrv_mean = window_df['HRV_RMSSD_ms'].mean()
    fatigue_mean = window_df['Estimated_Fatigue'].mean()

    check(hr_mean > adjust(0.9 * fc_max),
          "Heart rate critically high",
          ["Reduce pace immediately", "Switch to recovery mode", "Monitor breathing"])

    check(hrv_mean < adjust(adaptive_thresholds["HRV_RMSSD_ms"]),
          "HRV very low (high stress or fatigue)",
          ["Pause and breathe deeply", "Lower intensity", "Reassess training load"])

    check(fatigue_mean > adjust(adaptive_thresholds["Estimated_Fatigue"]),
          "Excessive fatigue accumulation",
          ["Stop training if symptoms appear", "Hydrate and recover", "Consider shortening next session"])

    # --------- FACTORES TÉRMICOS Y DE HIDRATACIÓN ----------
    temp_mean = window_df['Temperature_C'].mean()
    sweat_mean = window_df['Sweat_Loss_ml_min'].mean()
    elec_mean = window_df['Electrolyte_Loss_mmol_L'].mean()
    ve_mean = window_df['Ventilation_L_min'].dropna().mean()

    check(temp_mean > adjust_temp(adaptive_thresholds["Temperature_C"]),
          "Core temperature abnormally high",
          ["Cool down immediately", "Drink cold fluids", "Seek shade or ventilation"])

    check(sweat_mean > adjust(adaptive_thresholds["Sweat_Loss_ml_min"]),
          "Excessive sweating rate",
          ["Rehydrate frequently", "Add electrolytes", "Monitor thermal discomfort"])

    check(elec_mean > adjust(adaptive_thresholds["Electrolyte_Loss_mmol_L"]),
          "High electrolyte loss detected",
          ["Replenish with isotonic drinks", "Avoid water-only rehydration", "Pause if dizziness appears"])

    if not np.isnan(ve_mean):
        check(ve_mean > adjust(adaptive_thresholds["Ventilation_L_min"]),
              "Ventilation rate elevated",
              ["Control breathing rhythm", "Check posture and upper body relaxation"])

    # --------- FACTORES DE MOVIMIENTO ----------
    cadence_mean = window_df['Cadence_steps_min'].mean()
    impact_mean = window_df['Impact_g'].mean()

    check(cadence_mean < adaptive_thresholds["Cadence_steps_min"],
      "Cadence unusually low",
      ["Avoid overstriding", "Increase step frequency", "Check form"])

    check(impact_mean > adjust(adaptive_thresholds["Impact_g"]),
          "Muscular impact too high",
          ["Soften landings", "Shorten stride", "Use cushioned footwear"])

    # --------- ASIMETRÍA ARTICULAR ----------
    def check_asymmetry(left, right, label):
        if not left.empty and not right.empty:
            diff = np.abs(left.mean() - right.mean())
            rel_diff = diff / max(left.mean(), right.mean())
            if rel_diff > 0.25:
                check(True,
                      f"Asymmetry in {label} acceleration ({rel_diff*100:.1f}%)",
                      [f"Correct movement imbalance in {label.lower()}",
                       "Focus on bilateral coordination",
                       "Consult biomechanical expert if persistent"])

    check_asymmetry(window_df['Knee_Acceleration_Left_g'].dropna(),
                    window_df['Knee_Acceleration_Right_g'].dropna(),
                    "Knee")

    check_asymmetry(window_df['Ankle_Acceleration_Left_g'].dropna(),
                    window_df['Ankle_Acceleration_Right_g'].dropna(),
                    "Ankle")

    return ("🛑 Risk Factors Detected", alerts, actions) if alerts else ("✅ Parameters stable", [], [])








# ----------------------
# TRAINING IN PROGRESS
# ----------------------

if st.session_state.start and st.session_state.data is not None and not st.session_state.get('block_training_loop', False):
    df = st.session_state.data
    st.subheader("Live Training")
    risk_box = st.empty()
    live_box = st.empty()
    window_size = 30  # seconds
    if duration > st.session_state.duration:
        previous_data = st.session_state.data.iloc[:st.session_state.index + 1].copy()
        new_data = simulate_training(user, duration_min=duration)
        new_data = new_data.iloc[st.session_state.index + 1:].reset_index(drop=True)
        new_data['Time_s'] += previous_data['Time_s'].iloc[-1] + 1
        st.session_state.data = pd.concat([previous_data, new_data], ignore_index=True)
        st.session_state.duration = duration
        df = st.session_state.data

    def count_spikes(series, threshold):
        diffs = np.abs(np.diff(series))
        return np.sum(diffs > threshold)
    

    if 'last_messages' not in st.session_state:
        st.session_state.last_messages = [] 

    if 'prev_phase' not in st.session_state:
        st.session_state.prev_phase = -1
    phase_box = st.empty()
    while st.session_state.index < len(df):
        row = df.iloc[st.session_state.index]
        
        # Injury Risk Report
        df = st.session_state.data
        start_idx = max(0, st.session_state.index - window_size)
        df_row = df.iloc[st.session_state.index]
        recent_window = df.iloc[start_idx:st.session_state.index + 1]
        risk_status, messages, actions = evaluate_injury_risk(recent_window, user)
        if messages:
            st.session_state.alert_history.append((messages, actions))
            st.session_state.alert_timestamps.append(df_row["Time_s"])

        # Solo si cambian los mensajes mostramos nueva alerta
        if messages != st.session_state.last_messages:
            st.session_state.last_messages = messages
            st.session_state.messages = messages
            st.session_state.risk_status = risk_status

            status_color = 'red' if messages else 'green'
            risk_html = f"""
                <div id='risk-box' style='padding:1rem;background-color:#1E1E1E;border-radius:10px; color:#FF6B00; font-weight:500; margin-bottom:8px; animation: {'pulse 1s infinite' if messages else 'none'};' >
                <b>Injury Risk Monitor:</b> <span style='color:{status_color}'><b>{risk_status}</b></span><br>
                {('<br>'.join(messages)) if messages else 'All values within safe range.'}
                {"<hr style='margin-top:8px'><b>Clinical Recommendations:</b>" if actions else ''}
                {"<ul style='padding-left:1.2rem; line-height:1.6; font-size:0.95rem;'>" + ''.join([f"<li>{a}</li>" for a in actions]) + "</ul>" if actions else ''}
                </div>
                """
            risk_box.markdown(risk_html, unsafe_allow_html=True)

            # Sonido de alerta solo si hay nueva alerta
            st.markdown("""
            <audio autoplay>
            <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
            </audio>
            """, unsafe_allow_html=True)

        # Actualizar el live_box con parámetros fisiológicos y spikes
        live_box.markdown(f"""
        <div style='display:flex; gap:2rem; justify-content:center;'>
            <div style='width:48%; padding:1rem; background-color:#1E1E1E; border-radius:10px; color:#FF6B00; font-weight:500'>
                <b>Time:</b> {int(row['Time_s'])} s<br>
                <b>Heart Rate:</b> {int(row['Heart_Rate_bpm'])} bpm<br>
                <b>HRV:</b> {int(row['HRV_RMSSD_ms'])} ms<br>
                <b>Temperature:</b> {round(row['Temperature_C'], 1)} °C<br>
                <b>Fatigue:</b> {round(row['Estimated_Fatigue'], 2)}<br>
                <b>Sweat Loss:</b> {round(row['Sweat_Loss_ml_min'], 2)} ml/min<br>
                <b>Electrolyte Loss:</b> {round(row['Electrolyte_Loss_mmol_L'], 2)} mmol/L<br>
            </div>
            <div style='width:48%; padding:1rem; background-color:#1E1E1E; border-radius:10px; color:#FF6B00; font-weight:500'>
                <b>Cadence:</b> {int(row['Cadence_steps_min'])} steps/min<br>
                <b>Impact:</b> {round(row['Impact_g'], 2)} g<br>
                <b>Knee L Spikes:</b> {count_spikes(df['Knee_Acceleration_Left_g'].dropna()[:st.session_state.index+1], 0.5)} |
                <b>R:</b> {count_spikes(df['Knee_Acceleration_Right_g'].dropna()[:st.session_state.index+1], 0.5)}<br>
                <b>Ankle L Spikes:</b> {count_spikes(df['Ankle_Acceleration_Left_g'].dropna()[:st.session_state.index+1], 0.6)} |
                <b>R:</b> {count_spikes(df['Ankle_Acceleration_Right_g'].dropna()[:st.session_state.index+1], 0.6)}<br>
                {"<b>Ventilation:</b> " + str(round(row['Ventilation_L_min'], 1)) + " L/min<br>" if not np.isnan(row['Ventilation_L_min']) else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)
        phase_labels = ['Warm-up', 'Steady', 'Intense', 'Peak', 'Recovery', 'Cooldown']
        current_phase = int(row['Phase'])

        if current_phase != st.session_state.prev_phase:
            st.session_state.prev_phase = current_phase
            phase_box.markdown(
        f"""
        <div style='text-align:center; margin-top:1rem; margin-bottom:0.5rem;'>
            <span style='font-size:1.4rem; font-weight:700; color:#00FF7F;'>Current Phase: {phase_labels[current_phase]}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
        
        time.sleep(0.05)
        st.session_state.index += 1


# ----------------------
# PAST TRAININGS
# ----------------------

if not st.session_state.start:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Previous Trainings")
    folder = "trainings"
    os.makedirs(folder, exist_ok=True)
    files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")], reverse=True)

    if files:
        show_name = lambda f: f"★ {f}" if f in st.session_state.favorites else f
        selected = st.sidebar.selectbox("Select a training", ["None"] + [show_name(f) for f in files])
        selected_clean = selected.replace("★ ", "") if selected != "None" else None

        if selected_clean:
            col1, col2 = st.sidebar.columns(2)
            if col1.button("🗑 Delete"):
                os.remove(os.path.join(folder, selected_clean))
                st.sidebar.success("Deleted successfully.")
                st.stop()
            if selected_clean in st.session_state.favorites:
                if col2.button("✖ Unfavorite"):
                    st.session_state.favorites.remove(selected_clean)
            else:
                if col2.button("★ Favorite"):
                    st.session_state.favorites.add(selected_clean)

            st.subheader(f"Training: {'★ ' if selected_clean in st.session_state.favorites else ''}{selected_clean}")
            df_prev = pd.read_csv(os.path.join(folder, selected_clean))
            display_summary(df_prev)
    else:
        st.sidebar.write("No trainings available.")
