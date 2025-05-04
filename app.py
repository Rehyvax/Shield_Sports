import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

if 'start' not in st.session_state:
    st.session_state.start = False

if not st.session_state.start:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo.png", width=260)
        st.markdown("<h1 style='text-align:center; margin-top:0.2rem;'>Shield_Sports</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:gray;'>Injury prevention assistant</p>", unsafe_allow_html=True)
    st.markdown("---")

for key in ["start", "index", "data", "locked_user", "favorites"]:
    if key not in st.session_state:
        st.session_state[key] = set() if key == "favorites" else False if key == "start" else 0 if key == "index" else None
# ----------------------
# UTILITY FUNCTIONS
# ----------------------

def simulate_training(user, duration_min=10):
    t = np.arange(0, duration_min * 60, 1)
    age = user['age']
    gender = user['gender']
    level = user['level']
    fc_max = 220 - age - (5 if gender == 'female' else 0)

    if level == 'beginner':
        hr_base = (0.55, 0.70); hrv_base = (60, 100); cadence_avg = 160; impact_base = (1.2, 2.2)
    elif level == 'intermediate':
        hr_base = (0.60, 0.75); hrv_base = (40, 80); cadence_avg = 170; impact_base = (1.5, 2.5)
    else:
        hr_base = (0.50, 0.70); hrv_base = (30, 60); cadence_avg = 178; impact_base = (1.7, 2.8)

    hr = []; hrv = []; temp = []; cadence = []; impact = []; fatigue = []
    
    fat = 0.1
    fatigue_rate = {
    'beginner': 0.002,     # sube más rápido
    'intermediate': 0.0015,
    'advanced': 0.001      # sube más lento
    }[level]
    for i in t:
        phase = i // (duration_min * 60 // 5 + 1)
        noise = np.random.normal

        if phase == 0:
            hr_val = noise(loc=0.5*np.mean(hr_base)*fc_max, scale=3)
            hrv_val = noise(loc=np.mean(hrv_base)+10, scale=4)
            fat += fatigue_rate *0.3  # calentamiento: leve aumento
        elif phase in [1, 2]:
            hr_val = noise(loc=np.mean(hr_base)*fc_max, scale=2.5)
            hrv_val = noise(loc=np.mean(hrv_base), scale=5)
            fat += fatigue_rate *0.6  # trabajo moderado: subida estable
        elif phase == 3:
            hr_val = noise(loc=0.9*np.mean(hr_base)*fc_max, scale=4)
            hrv_val = noise(loc=np.mean(hrv_base)-10, scale=6)
            fat += fatigue_rate * 1.2  # pico de esfuerzo: subida rápida
        elif phase == 4:  # Recuperación
            hr_val = noise(loc=0.6*np.mean(hr_base)*fc_max, scale=2)
            hrv_val = noise(loc=np.mean(hrv_base)+5, scale=5)
            fat -= fatigue_rate *0.7  # la fatiga disminuye de forma activa
        else:
            hr_val = noise(loc=0.7*np.mean(hr_base)*fc_max, scale=3)
            hrv_val = noise(loc=np.mean(hrv_base), scale=7)
            fat += fatigue_rate *0.4


        fat = np.clip(fat, 0, 1)

        cad_val = noise(loc=cadence_avg, scale=1.5)
        imp_val = noise(loc=np.mean(impact_base), scale=0.1) + 0.0003 * i
        temp_val = 36.5 + 0.01*(i/60) + noise(0, 0.05)

        hr.append(hr_val)
        hrv.append(np.clip(hrv_val, 20, 120))
        temp.append(temp_val)
        cadence.append(cad_val)
        impact.append(imp_val)
        fatigue.append(fat)

    total_time = duration_min * 60
    borders = np.cumsum([0, 0.10, 0.40, 0.20, 0.10, 0.20]) * total_time
    phases = np.zeros_like(t)
    for i, (start, end) in enumerate(zip(borders[:-1], borders[1:])):
        phases[(t >= start) & (t < end)] = i

    knee_acc = []
    ankle_acc = []

    if st.session_state.get('knee_band'):
        knee_acc = list(np.random.normal(1.4, 0.15, len(t)) + 0.001*np.array(t))
    else:
        knee_acc = [np.nan]*len(t)

    if st.session_state.get('ankle_band'):
        ankle_acc = list(np.random.normal(1.8, 0.2, len(t)) + 0.0015*np.array(t))
    else:
        ankle_acc = [np.nan]*len(t)

    return pd.DataFrame({
        'Time_s': t,
        'Heart_Rate_bpm': hr,
        'HRV_RMSSD_ms': hrv,
        'Temperature_C': temp,
        'Cadence_steps_min': cadence,
        'Impact_g': impact,
        'Estimated_Fatigue': fatigue,
        'Phase': phases,
        'Knee_Acceleration_g': knee_acc,
        'Ankle_Acceleration_g': ankle_acc
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
<div id='risk-box' style='padding:1rem;background-color:#f0f0f0;border-radius:10px;margin-bottom:8px; animation: {'pulse 1s infinite' if messages else 'none'};'>
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
    st.subheader("Training Summary")
    means = df.drop(columns=["Time_s", "Phase"]).mean().round(2)
    cols = st.columns(3)
    for i, (param, value) in enumerate(means.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div style='background-color:#f9f9f9;padding:15px;border-radius:10px;border:1px solid #e0e0e0;text-align:center'>
                <div style='font-size:14px;color:#777;'>{param.replace('_', ' ')}</div>
                <div style='font-size:20px;font-weight:bold;color:#333'>{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Training Parameter Evolution")
    fig, ax = plt.subplots(figsize=(10, 4))
    phase_colors = ['#e8f5e9', '#fffde7', '#ffebee', '#e3f2fd', '#ede7f6']
    phase_names = ['Warm-up', 'Steady', 'Intense', 'Peak', 'Recovery']
    times = df['Time_s'].values
    phases = df['Phase'].values
    for p in range(5):
        mask = phases == p
        if np.any(mask):
            ax.axvspan(times[mask][0], times[mask][-1], color=phase_colors[p], alpha=0.45)
    for col in ["Heart_Rate_bpm", "HRV_RMSSD_ms", "Cadence_steps_min", "Knee_Acceleration_g", "Ankle_Acceleration_g"]:
        ax.plot(df["Time_s"], df[col], label=col.replace('_', ' '), linewidth=1.5)

    # Display variation summary for low-variation parameters
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("Training Parameters Over Time", fontsize=13, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9)
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
    st.markdown("<div style='display: flex; justify-content: space-around; margin-top: 20px;'>", unsafe_allow_html=True)
    for var in ["Temperature_C", "Impact_g", "Estimated_Fatigue"]:
            start_val = round(df[var].iloc[0], 2)
            end_val = round(df[var].iloc[-1], 2)
            st.markdown(f"""
            <div style='background-color:#fdfdfd;padding:10px 15px;border-radius:8px;border:1px solid #ddd;min-width:150px;text-align:center;'>
                <span style='color:#555;font-weight:600'>{var.replace('_', ' ')}</span><br>
                <span style='color:#333'>{start_val} → {end_val}</span>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# INJURY REPORT FUNCTION
# ----------------------

def generate_injury_report(df, user):
    st.subheader("Injury Risk Report")
    fc_max = 220 - user['age'] - (5 if user['gender'] == 'female' else 0)
    alerts = []
    suggestions = []

    def check(param, mean_val, threshold, direction, message, advice):
        if (direction == '>' and mean_val > threshold) or (direction == '<' and mean_val < threshold):
            alerts.append(f"⚠️ {message}: {mean_val:.2f}")
            suggestions.append(advice)

    mean_hr = df['Heart_Rate_bpm'].mean()
    mean_hrv = df['HRV_RMSSD_ms'].mean()
    mean_impact = df['Impact_g'].mean()
    mean_fatigue = df['Estimated_Fatigue'].mean()

    check('HR', mean_hr, 0.9 * fc_max, '>', "High Heart Rate", "Reduce intensity or slow down.")
    check('HRV', mean_hrv, 30, '<', "Low HRV (stress/fatigue)", "Apply breathing control, lower intensity.")
    check('Impact', mean_impact, 2.5, '>', "High Muscular Impact", "Consider softening your landing or adjusting technique.")
    check('Fatigue', mean_fatigue, 0.75, '>', "High Fatigue Load", "Shift to recovery pace or stop if needed.")

    if 'Knee_Acceleration_g' in df.columns:
        knee_acc = df['Knee_Acceleration_g'].dropna()
        if not knee_acc.empty:
            mean_knee = knee_acc.mean()
            check('Knee', mean_knee, 1.8, '>', "High Knee Acceleration", "Adjust extension pattern or stride.")

    if 'Ankle_Acceleration_g' in df.columns:
        ankle_acc = df['Ankle_Acceleration_g'].dropna()
        if not ankle_acc.empty:
            mean_ankle = ankle_acc.mean()
            check('Ankle', mean_ankle, 2.0, '>', "High Ankle Acceleration", "Review footstrike and avoid excessive torsion.")

    if alerts:
        st.error("Risk Factors Detected:")
        for a in alerts:
            st.markdown(f"- {a}")
        st.warning("Suggested Actions:")
        for s in suggestions:
            st.markdown(f"- {s}")
    else:
        st.success("All measured parameters remained within physiological ranges. No significant risk detected.")


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


# ----------------------

st.sidebar.title("User Profile")
if not st.session_state.start:
    age = st.sidebar.slider("Age", 16, 75, 30)
    gender = st.sidebar.selectbox("Gender", ["male", "female"])
    weight = st.sidebar.slider("Weight (kg)", 40, 120, 70)
    level = st.sidebar.selectbox("Level", ["beginner", "intermediate", "advanced"])
    user = {"age": age, "gender": gender, "weight": weight, "level": level}
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
    st.rerun()

if st.sidebar.button("⏹ Stop Training"):
    st.session_state.start = False
    if st.session_state.data is not None:
        folder = "trainings"
        os.makedirs(folder, exist_ok=True)
        df_save = st.session_state.data.iloc[:st.session_state.index]
        if not df_save.empty:
            filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
            path = os.path.join(folder, filename)
            df_save.to_csv(path, index=False)
            st.success(f"Training saved as {filename}")
            display_summary(df_save)
            generate_injury_report(df_save, user)

# ----------------------
# INJURY RISK ASSESSMENT FUNCTION
# ----------------------

def evaluate_injury_risk(window_df, user):
    fc_max = 220 - user['age'] - (5 if user['gender'] == 'female' else 0)
    alerts = []
    actions = []

    def check(cond, message, advice_list):
        if cond:
            alerts.append(f"⚠️ {message}")
            actions.extend(advice_list)

    check(window_df['Heart_Rate_bpm'].mean() > 0.9 * fc_max,
          "High heart rate",
          ["Reduce your pace immediately.",
           "Focus on deep, rhythmic breathing.",
           "Switch to walking until heart rate lowers."])

    check(window_df['HRV_RMSSD_ms'].mean() < 30,
          "Very low HRV",
          ["Take deep diaphragmatic breaths.",
           "Avoid sudden changes in pace.",
           "Pause to let HRV recover."])

    check(window_df['Impact_g'].mean() > 2.5,
          "High impact load",
          ["Land more softly.",
           "Shorten your stride.",
           "Avoid bouncing steps."])

    check(window_df['Estimated_Fatigue'].mean() > 0.8,
          "Fatigue is critically high",
          ["Lower training intensity.",
           "Hydrate and rest.",
           "Consider switching to recovery mode."])

    phase = window_df['Phase'].iloc[-1] if 'Phase' in window_df.columns else None

    def count_spikes(series, threshold):
        diffs = np.abs(np.diff(series))
        return np.sum(diffs > threshold)

    if 'Knee_Acceleration_g' in window_df.columns:
        knee = window_df['Knee_Acceleration_g'].dropna()
        if not knee.empty and phase == 3:
            if knee.mean() > 1.8:
                check(True, "High knee acceleration",
                      ["Reduce jump intensity.",
                       "Control lateral knee motion.",
                       "Avoid sudden deceleration."])
            spikes_knee = count_spikes(knee, 0.5)
            if spikes_knee > 5:
                check(True, f"Abrupt knee acceleration changes ({spikes_knee})",
                      ["Avoid rapid changes of pace.",
                       "Stabilize knee alignment.",
                       "Try to maintain smoother strides."])

    if 'Ankle_Acceleration_g' in window_df.columns:
        ankle = window_df['Ankle_Acceleration_g'].dropna()
        if not ankle.empty and phase == 3:
            if ankle.mean() > 2.0:
                check(True, "High ankle acceleration",
                      ["Avoid heel striking too hard.",
                       "Focus on mid-foot landing.",
                       "Stabilize ankle joints."])
            spikes_ankle = count_spikes(ankle, 0.6)
            if spikes_ankle > 5:
                check(True, f"Abrupt ankle acceleration changes ({spikes_ankle})",
                      ["Avoid sharp directional changes.",
                       "Keep cadence more consistent.",
                       "Maintain stable ground contact."])

    return ("🛑 Potential Injury Risk", alerts, actions) if alerts else ("✅ Parameters stable", [], [])





# ----------------------
# TRAINING IN PROGRESS
# ----------------------

if st.session_state.start and st.session_state.data is not None:
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

    while st.session_state.index < len(df):
        row = df.iloc[st.session_state.index]
        
        # Injury Risk Report
        start_idx = max(0, st.session_state.index - window_size)
        recent_window = df.iloc[start_idx:st.session_state.index + 1]
        risk_status, messages, actions = evaluate_injury_risk(recent_window, user)

        # Solo si cambian los mensajes mostramos nueva alerta
        if messages != st.session_state.last_messages:
            st.session_state.last_messages = messages
            st.session_state.messages = messages
            st.session_state.risk_status = risk_status

            status_color = 'red' if messages else 'green'
            risk_html = f"""
                <div id='risk-box' style='padding:1rem;background-color:#f0f0f0;border-radius:10px;margin-bottom:8px; animation: {'pulse 1s infinite' if messages else 'none'};'>
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
        <div style='padding:1rem;background-color:#f0f0f0;border-radius:10px'>
            <b>Time:</b> {int(row['Time_s'])} s<br>
            <b>Heart Rate:</b> {int(row['Heart_Rate_bpm'])} bpm<br>
            <b>HRV:</b> {int(row['HRV_RMSSD_ms'])} ms<br>
            <b>Temp:</b> {round(row['Temperature_C'], 1)} °C<br>
            <b>Cadence:</b> {int(row['Cadence_steps_min'])} steps/min<br>
            <b>Impact:</b> {round(row['Impact_g'], 2)} g<br>
            <b>Fatigue:</b> {round(row['Estimated_Fatigue'], 2)}<br>
            <b>Knee Acc:</b> {round(row['Knee_Acceleration_g'], 2) if not np.isnan(row['Knee_Acceleration_g']) else 'N/A'} g<br>
            <b>Ankle Acc:</b> {round(row['Ankle_Acceleration_g'], 2) if not np.isnan(row['Ankle_Acceleration_g']) else 'N/A'} g<br>
            <hr style='margin: 5px 0'>
            <b>Knee Spikes:</b> {count_spikes(df['Knee_Acceleration_g'].dropna()[:st.session_state.index+1], 0.5)}<br>
            <b>Ankle Spikes:</b> {count_spikes(df['Ankle_Acceleration_g'].dropna()[:st.session_state.index+1], 0.6)}<br>
        </div>
        """, unsafe_allow_html=True)

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
