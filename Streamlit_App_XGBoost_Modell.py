import streamlit as st 
import pandas as pd
import joblib
import numpy as np 

st.set_page_config(layout="wide")
result_container = st.sidebar.container()
st.markdown("<h1 style='text-align: center; margin-top: 40px;'>🔍 Kreditrisiko-Vorhersage</h1>", unsafe_allow_html=True)
st.write("Diese Anwendung hilft dem Vertriebsteam dabei, die Abschlusswahrscheinlichkeit einer Kredit vorherzusagen.")

# Model und Preprocessing laden
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("xgb_credit_risk.pkl")

# Daten laden
df = pd.read_csv("GermanCredit_eur.csv", sep=',')

def main():
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Erste Zeile: Titel und erklärender Text
    st.header("📊 Kreditrisiko-Prognose")
    st.markdown("Geben Sie die Kreditdaten ein, um das Risiko zu berechnen. Unser Modell hilft Ihnen, eine fundierte Entscheidung zu treffen.")

    # Erstellen Sie zwei Spalten
    col1, col2 = st.columns([1, 1])  # Gleiche Breite für beide Spalten
    
    with col1:
        st.subheader("📝 Eingabedaten")
        # Eingabefelder in der ersten Spalte
        status = st.selectbox("Kontostatus", df['status'].unique().tolist())
        credit_history = st.selectbox("Kreditverlauf", df['credit_history'].unique().tolist())
        purpose = st.selectbox("Kreditverwendungsaweck", df['purpose'].unique().tolist())
        savings = st.selectbox("Sparguthaben", df['savings'].unique().tolist())
        employment_duration = st.selectbox("Beschäftigungsdauer", df["employment_duration"].unique().tolist())
        personal_status_sex = st.selectbox("Familienstand/Geschlecht", df["personal_status_sex"].unique().tolist())  # Fixed here
        other_debtors = st.selectbox("Weitere Schuldner", df['other_debtors'].unique().tolist())
        property = st.selectbox("Besitz", df["property"].unique().tolist())
        other_installment_plans = st.selectbox("Weitere Ratenzahlungen", df["other_installment_plans"].unique().tolist())
        housing = st.selectbox("Wohnsituation", df["housing"].unique().tolist())
        job = st.selectbox("Beruf", df["job"].unique().tolist())
        foreign_worker = st.selectbox("Ausländischer Arbeiter", ["yes", "no"])
    
    with col2:    
        st.subheader("🔧 Kreditparameter")
        # Weitere Eingabefelder in der zweiten Spalte
        duration = st.slider("Laufzeit des Kredits (Monate)", 4, 72, 6)
        amount = st.slider("Kredithöhe (DM)", 1000, 100000, 2000)
        installment_rate = st.slider("Ratenhöhe (% des Einkommens)", 1, 4, 2)
        present_residence = st.slider("Jahre im aktuellen Wohnsitz", 1, 4, 2)
        age = st.slider("Alter (Jahre)", 18, 100, 30)
        number_credits = st.slider("Anzahl der Kredite", 1, 4, 1)
        people_liable = st.slider("Anzahl unterhaltsberechtigter Personen", 1, 2, 1)

    # Styling für den Eingabebereich
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
        }
        </style>
        """, unsafe_allow_html=True)

    # Erstellen Sie ein DataFrame mit den Benutzereingaben
    input_data = pd.DataFrame([{
        'status': status,
        'duration': duration,
        'credit_history': credit_history,
        'purpose': purpose,
        'amount': amount,
        'savings': savings,
        'employment_duration': employment_duration,
        'installment_rate': installment_rate,
        'personal_status_sex': personal_status_sex,
        'other_debtors': other_debtors,
        'present_residence': present_residence,
        'property': property,
        'age': age,
        'other_installment_plans': other_installment_plans,
        'housing': housing,
        'number_credits': number_credits,
        'job': job,
        'people_liable': people_liable,
        'foreign_worker': foreign_worker
    }])  
    
    # Daten transformieren
    input_transformed = preprocessor.transform(input_data)  # No reshape needed
    
    # Vorhersage
    prediction = model.predict(input_transformed) 
    prediction_probability = model.predict_proba(input_transformed)[0][1]
    
    # Wenn die Wahrscheinlichkeit irgendwie über 100% geht, normalisieren wir sie
    if prediction_probability > 1:
        prediction_probability = 1.0
    
    # Visualisierung der Ergebnisse
    
    if prediction == 1:
        with result_container:
            st.subheader("🔍 Vorhersage-Ergebnisse")
            st.error(f"❌ Hohes Kreditrisiko! (Wahrscheinlichkeit: {prediction_probability:.2%})", icon="🚨")
    else:
        with result_container:
            st.success(f"✅ Geringes Kreditrisiko! (Wahrscheinlichkeit: {(100 - prediction_probability)/100:.2%})", icon="✔️")
        
    # Weitere Hilfe- oder Erklärungshinweise
    result_container.markdown("---")
    result_container.info("Diese Anwendung wurde entwickelt, um Ihnen zu helfen, das Kreditrisiko vorherzusagen. Es basiert auf maschinellen Lernmodellen, die mit historischen Kreditdaten trainiert wurden.")
        
# Streamlit-App starten
if __name__ == "__main__":
    main()    