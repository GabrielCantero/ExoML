import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ExoML Dashboard", layout="wide")
st.title("🌌 ExoML: Detección de Exoplanetas")

# 1️⃣ Cargar dataset
@st.cache_data
def cargar_datos():
    df = pd.read_csv("cumualative.csv")  # tu archivo CSV
    features = ["koi_prad", "koi_srad", "koi_period", "koi_teq"]
    target = "koi_disposition"
    df = df[features + [target]]
    df = df.dropna()
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])
    return df, features, target, le

df, features, target, le = cargar_datos()

# 2️⃣ Entrenar modelo
@st.cache_data
def entrenar_modelo(df, features, target):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = entrenar_modelo(df, features, target)

# 3️⃣ Panel lateral para predecir un nuevo planeta
st.sidebar.header("Predecir nuevo planeta")
koi_prad = st.sidebar.slider("Radio del planeta (tierras)", 0.1, 20.0, 1.0)
koi_srad = st.sidebar.slider("Radio de la estrella (soles)", 0.1, 3.0, 1.0)
koi_period = st.sidebar.slider("Periodo orbital (días)", 0.1, 500.0, 100.0)
koi_teq = st.sidebar.slider("Temperatura equilibrio (K)", 100, 1000, 300)

nuevo_planeta = pd.DataFrame({
    "koi_prad": [koi_prad],
    "koi_srad": [koi_srad],
    "koi_period": [koi_period],
    "koi_teq": [koi_teq]
})
prediccion = model.predict(nuevo_planeta)
etiqueta = le.inverse_transform(prediccion)[0]

st.sidebar.markdown(f"**Predicción:** {etiqueta}")

# 4️⃣ Gráfico 1: Cantidad de planetas por clase
st.subheader("Cantidad de planetas por clase")
counts = df[target].value_counts()
labels = le.inverse_transform(counts.index)
fig1, ax1 = plt.subplots()
sns.barplot(x=labels, y=counts.values, palette="viridis", ax=ax1)
ax1.set_ylabel("Número de planetas")
st.pyplot(fig1)

# 5️⃣ Gráfico 2: Radio vs periodo orbital
st.subheader("Distribución de planetas por radio y periodo")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x="koi_period", y="koi_prad", hue=target, palette="viridis", alpha=0.7, ax=ax2)
ax2.set_xlabel("Periodo orbital (días)")
ax2.set_ylabel("Radio del planeta (tierras)")
ax2.legend(labels=le.inverse_transform([0,1,2]))
st.pyplot(fig2)
