import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Title of the Streamlit App
st.title('Analisis Penyewaan Sepeda')

# Load Dataset
url = "https://raw.githubusercontent.com/staryzcode/dicoding-task/main/day.csv"
customers_df = pd.read_csv(url)

# Display raw data
st.subheader('Data Penyewaan Sepeda')
st.write(customers_df.head())

# Data Description
st.subheader('Assessing Data')
st.write(customers_df.describe())

# EDA: Group by Holiday
st.subheader('Exploratory Data Analysis (EDA): Pengelompokan Berdasarkan Hari Libur')
holiday_group = customers_df.groupby(by=["holiday"]).agg({
    "cnt": ["max", "min", "mean", "std"],
}).reset_index()

st.write('Tabel: Rangkuman Penyewaan Sepeda Berdasarkan Hari Libur')
st.write(holiday_group)

# EDA: Group by Weather
st.subheader('Exploratory Data Analysis (EDA): Pengelompokan Berdasarkan Kondisi Cuaca')
weather_group = customers_df.groupby(by=["weathersit"]).agg({
    "cnt": ["max", "min", "mean", "std"],
}).reset_index()

st.write('Tabel: Rangkuman Penyewaan Sepeda Berdasarkan Kondisi Cuaca')
st.write(weather_group)

# Visualization 1: Distribusi Penyewaan Sepeda Berdasarkan Hari Libur
st.subheader('Distribusi Penyewaan Sepeda pada Hari Libur dan Non-Libur')
fig, ax = plt.subplots()
sns.boxplot(data=customers_df, x='holiday', y='cnt', ax=ax)
ax.set_title('Distribusi Penyewaan Sepeda pada Hari Libur dan Non-Libur')
ax.set_xlabel('Hari Libur (0 = Non-Libur, 1 = Libur)')
ax.set_ylabel('Total Penyewaan Sepeda')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Non-Libur', 'Libur'])
plt.grid(True)
st.pyplot(fig)

# Visualization 2: Pengaruh Kondisi Cuaca terhadap Penyewaan Sepeda
st.subheader('Pengaruh Kondisi Cuaca terhadap Penyewaan Sepeda')
weather_group = customers_df.groupby('weathersit')['cnt'].mean().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=weather_group, x='weathersit', y='cnt', ax=ax)
ax.set_title('Pengaruh Kondisi Cuaca terhadap Jumlah Penyewaan Sepeda')
ax.set_xlabel('Kondisi Cuaca (weathersit)')
ax.set_ylabel('Rata-rata Penyewaan Sepeda')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['1: Cerah', '2: Kabut', '3: Hujan '])
plt.grid(True)
st.pyplot(fig)

# Advanced Analysis: Korelasi Antar Variabel
st.subheader('Analisis Lanjutan: Korelasi Antar Variabel')
corr_matrix = customers_df[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].corr()
fig, ax = plt.subplots()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Heatmap Korelasi Antar Variabel')
st.pyplot(fig)

# Menjelaskan Korelasi
st.write("""
Dari heatmap korelasi di atas, kita bisa melihat hubungan antara beberapa variabel seperti suhu (`temp`), suhu yang dirasakan (`atemp`), kelembapan (`hum`), kecepatan angin (`windspeed`), dan jumlah penyewaan sepeda (`cnt`). 
Nilai korelasi berkisar antara -1 hingga 1:
- Nilai mendekati 1 menunjukkan korelasi positif yang kuat (jika x meningkat maka y akan meningkat).
- Nilai mendekati -1 menunjukkan korelasi negatif yang kuat (jika x menurun maka y akan menurun).
- Nilai mendekati 0 menunjukkan tidak ada korelasi.
""")
