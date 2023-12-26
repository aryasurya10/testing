import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans


@st.cache
def load_data():
    data = pd.read_csv('Mall_Customers.csv')
    return data


def main():
    st.title('Aplikasi Data Mining dengan Metode K-Means')
    st.write('Menggunakan Streamlit')

    data = load_data()

    st.subheader('Dataset')
    st.write(data)

    st.subheader('Hasil Clustering dengan K-Means')
    k = st.slider('Jumlah Cluster (K)', 2, 10, 3)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    data['Cluster'] = kmeans.labels_
    st.write(data)


if __name__ == '__main__':
    main()
