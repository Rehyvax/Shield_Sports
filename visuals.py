from pathlib import Path

# Crear el contenido mejorado de visuals.py
import plotly.graph_objects as go
import streamlit as st

def display_hr_gauge(value, fc_max):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'font': {'size': 28}},
        gauge={
            'axis': {'range': [40, fc_max + 10]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [40, 0.6 * fc_max], 'color': "#8BC34A"},
                {'range': [0.6 * fc_max, 0.8 * fc_max], 'color': "#FFC107"},
                {'range': [0.8 * fc_max, fc_max + 10], 'color': "#F44336"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

def display_temp_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'font': {'size': 28}},
        gauge={
            'axis': {'range': [35, 41]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [35, 37.5], 'color': "#8BC34A"},
                {'range': [37.5, 38.5], 'color': "#FFC107"},
                {'range': [38.5, 41], 'color': "#F44336"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

def display_hrv_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'font': {'size': 28}},
        gauge={
            'axis': {'range': [10, 100]},
            'bar': {'color': "blue"},
            'steps': [
                {'range': [10, 30], 'color': "#F44336"},
                {'range': [30, 50], 'color': "#FFC107"},
                {'range': [50, 100], 'color': "#8BC34A"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

def display_fatigue_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'font': {'size': 28}},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "orange"},
            'steps': [
                {'range': [0, 0.4], 'color': "#8BC34A"},
                {'range': [0.4, 0.7], 'color': "#FFC107"},
                {'range': [0.7, 1], 'color': "#F44336"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

def display_sweat_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'font': {'size': 28}},
        gauge={
            'axis': {'range': [0, 5]},
            'bar': {'color': "blue"},
            'steps': [
                {'range': [0, 1.5], 'color': "#8BC34A"},
                {'range': [1.5, 3], 'color': "#FFC107"},
                {'range': [3, 5], 'color': "#F44336"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

def display_elec_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'font': {'size': 28}},
        gauge={
            'axis': {'range': [0, 0.2]},
            'bar': {'color': "blue"},
            'steps': [
                {'range': [0, 0.05], 'color': "#8BC34A"},
                {'range': [0.05, 0.1], 'color': "#FFC107"},
                {'range': [0.1, 0.2], 'color': "#F44336"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

def display_cadence_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'font': {'size': 28}},
        gauge={
            'axis': {'range': [130, 190]},
            'bar': {'color': "purple"},
            'steps': [
                {'range': [130, 150], 'color': "#F44336"},
                {'range': [150, 165], 'color': "#FFC107"},
                {'range': [165, 190], 'color': "#8BC34A"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

def display_impact_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'font': {'size': 28}},
        gauge={
            'axis': {'range': [0, 4]},
            'bar': {'color': "brown"},
            'steps': [
                {'range': [0, 1.8], 'color': "#8BC34A"},
                {'range': [1.8, 2.5], 'color': "#FFC107"},
                {'range': [2.5, 4], 'color': "#F44336"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)


