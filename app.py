import streamlit as st
import tempfile
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans

# =========================================
# Step 1: Start Spark & Streamlit
# =========================================
spark = SparkSession.builder \
    .appName("KMeans Clustering App") \
    .getOrCreate()

st.title("Web-Based K-Means Clustering System")

# =========================================
# Step 2: Upload and Load Data
# =========================================
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    # Read file using PySpark
    df = spark.read.csv(temp_path, header=True, inferSchema=True)

    # Fix column name 
    df = df.withColumnRenamed("Bytes Transferred", "Bytes_Transferred")

    # =========================================
    # Step 3: Preview Dataset
    # =========================================
    st.subheader("Dataset Preview")
    st.write(df.limit(5).toPandas())

    st.write("Rows:", df.count())
    st.write("Columns:", len(df.columns))

    # =========================================
    # Step 4: Prepare Data
    # =========================================
    # Select only numeric columns
    numeric_cols = [
        "Session_Connection_Time",
        "Bytes_Transferred",
        "Servers_Corrupted",
        "Pages_Corrupted",
        "WPM_Typing_Speed"
    ]

    df_clean = df.select(numeric_cols).dropna()

    st.subheader("Selected Features")
    st.write(numeric_cols)

    # =========================================
    # Step 5: Create Feature Vector
    # =========================================
    assembler = VectorAssembler(
        inputCols=numeric_cols,
        outputCol="features"
    )
    df_features = assembler.transform(df_clean)

    # =========================================
    # Step 6: Scale Features
    # =========================================
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features"
    )
    scaler_model = scaler.fit(df_features)
    df_scaled = scaler_model.transform(df_features)

    # =========================================
    # Step 7: Run K-Means Clustering
    # =========================================
    st.subheader("K-Means Clustering")

    k = st.slider("Select number of clusters (k)", 2, 5, 3)

    kmeans = KMeans(
        featuresCol="scaled_features",
        k=k
    )

    model = kmeans.fit(df_scaled)
    df_result = model.transform(df_scaled)

    # =========================================
    # Show Results
    # =========================================
    st.subheader("Clustering Results")
    st.write(df_result.select(numeric_cols + ["prediction"]).toPandas())

    # Cluster Centers
    st.subheader("Cluster Centers")
    centers = model.clusterCenters()
    st.write(centers)