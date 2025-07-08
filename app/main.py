import streamlit as st
import pickle
import pandas as pd
import emoji
import plotly.graph_objects as go
import numpy as np


def get_clean_data():
    data = pd.read_csv('data/data.csv')
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclie Measurements")

        # ➕ Instruction block
    st.sidebar.markdown("""
    <div style='font-size: 15px; line-height: 1.6; padding-bottom: 10px; color: #e1e1e1;'>
        🧪 <b>Instructions:</b><br>
        Use the sliders below to simulate various cell nucleus features extracted from breast tissue samples.<br>
        🧠 <i>The app will predict whether the tumor is likely <b>Benign</b> or <b>Malignant</b>.</i>
    </div>
    """, unsafe_allow_html=True)

    data = get_clean_data()

    slider_labels = [
    ("Radius (Mean)", "radius_mean"),
    ("Texture (Mean)", "texture_mean"),
    ("Perimeter (Mean)", "perimeter_mean"),
    ("Area (Mean)", "area_mean"),
    ("Smoothness (Mean)", "smoothness_mean"),
    ("Compactness (Mean)", "compactness_mean"),
    ("Concavity (Mean)", "concavity_mean"),
    ("Concave Points (Mean)", "concave points_mean"),
    ("Symmetry (Mean)", "symmetry_mean"),
    ("Fractal Dimension (Mean)", "fractal_dimension_mean"),

    ("Radius (SE)", "radius_se"),
    ("Texture (SE)", "texture_se"),
    ("Perimeter (SE)", "perimeter_se"),
    ("Area (SE)", "area_se"),
    ("Smoothness (SE)", "smoothness_se"),
    ("Compactness (SE)", "compactness_se"),
    ("Concavity (SE)", "concavity_se"),
    ("Concave Points (SE)", "concave points_se"),
    ("Symmetry (SE)", "symmetry_se"),
    ("Fractal Dimension (SE)", "fractal_dimension_se"),

    ("Radius (Worst)", "radius_worst"),
    ("Texture (Worst)", "texture_worst"),
    ("Perimeter (Worst)", "perimeter_worst"),
    ("Area (Worst)", "area_worst"),
    ("Smoothness (Worst)", "smoothness_worst"),
    ("Compactness (Worst)", "compactness_worst"),
    ("Concavity (Worst)", "concavity_worst"),
    ("Concave Points (Worst)", "concave points_worst"),
    ("Symmetry (Worst)", "symmetry_worst"),
    ("Fractal Dimension (Worst)", "fractal_dimension_worst")]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value = float(0),
            max_value = float(data[key].max()),
            value=float(data[key].mean())
        )

    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()

    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val)/(max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict

def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area',
                  'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'],
            input_data['area_se'], input_data['smoothness_se'], input_data['compactness_se'],
            input_data['concavity_se'], input_data['concave points_se'], input_data['symmetry_se'],
            input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Errors'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))


    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True
    )

    return fig

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scalar = pickle.load(open("model/scalar.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scalar.transform(input_array)

    prediction = model.predict(input_array_scaled)
    prob_benign = model.predict_proba(input_array_scaled)[0][0]
    prob_malignant = model.predict_proba(input_array_scaled)[0][1]

    st.subheader("🔬 Prediction Results")
    
    if prediction[0] == 0:
        st.markdown(
            "<div style='background-color:#d1e7dd;padding:8px;border-radius:6px'>"
            "<h4 style='color:#0f5132; text-align:center;'>🟢 <b>Benign</b></h4>"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background-color:#f8d7da;padding:8px;border-radius:6px'>"
            "<h4 style='color:#842029; text-align:center;'>🔴 <b>Malignant</b></h4>"
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.metric(label="Probability of Benign", value=f"{prob_benign:.2%}")
    st.metric(label="Probability of Malignant", value=f"{prob_malignant:.2%}")

    st.markdown(
        "<div style='font-size: 13px; color: grey; padding-top: 10px;'>"
        "⚠️ This tool provides an estimate using machine learning and is designed to aid learning and exploration. "
        "It should not be used for medical decision-making.</div>",
        unsafe_allow_html=True
    )

def main():
    st.set_page_config(
        page_title='Breast Cancer Predictor',
        page_icon= emoji.emojize(":stethoscope:"),
        layout='wide',
        initial_sidebar_state='expanded'
    )

    input_data = add_sidebar()

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("This app uses a machine learning model to predict whether a tumor is benign or malignant based on input features from a breast cancer dataset. The model is trained on real diagnostic data and helps demonstrate how AI can assist in early detection and medical decision-making.")
    
    col1, col2 = st.columns([4,1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)

    
if __name__ == '__main__':
    main()