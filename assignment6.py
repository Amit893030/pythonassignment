#Q1.a.Design a data ingestion pipeline that collects and stores data from various sources such as databases, APIs, and streaming platforms.
import requests
import json
import psycopg2
from kafka import KafkaProducer

# Step 1: Data Collection from APIs
def fetch_data_from_api(api_url):
    response = requests.get(api_url)
    data = response.json()
    return data

# Step 2: Data Extraction from Databases
def extract_data_from_database(db_connection_params, query):
    conn = psycopg2.connect(**db_connection_params)
    cursor = conn.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return data

# Step 3: Data Streaming
def stream_data_to_kafka(bootstrap_servers, topic, data):
    producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
    for record in data:
        producer.send(topic, json.dumps(record).encode('utf-8'))
    producer.flush()

# Step 4: Data Storage (Example: Writing to a File)
def write_data_to_file(file_path, data):
    with open(file_path, 'a') as file:
        for record in data:
            file.write(json.dumps(record) + '\n')

# Usage Example
if __name__ == '__main__':
    # Step 1: Fetch data from API
    api_url = 'https://api.example.com/data'
    api_data = fetch_data_from_api(api_url)

    # Step 2: Extract data from a database
    db_connection_params = {
        'host': 'localhost',
        'port': 5432,
        'database': 'mydatabase',
        'user': 'myuser',
        'password': 'mypassword'
    }
    query = 'SELECT * FROM mytable'
    db_data = extract_data_from_database(db_connection_params, query)

    # Step 3: Stream data to Kafka
    kafka_bootstrap_servers = ['localhost:9092']
    kafka_topic = 'mytopic'
    stream_data_to_kafka(kafka_bootstrap_servers, kafka_topic, api_data)

    # Step 4: Write data to a file
    file_path = 'data.txt'
    write_data_to_file(file_path, db_data)


#1b.. Implement a real-time data ingestion pipeline for processing sensor data from IoT device
from kafka import KafkaConsumer
import json
import psycopg2

# Step 1: Configure Kafka Consumer
kafka_bootstrap_servers = ['localhost:9092']
kafka_topic = 'sensor_data_topic'

consumer = KafkaConsumer(
    kafka_topic,
    bootstrap_servers=kafka_bootstrap_servers,
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Step 2: Configure Database Connection
db_connection_params = {
    'host': 'localhost',
    'port': 5432,
    'database': 'mydatabase',
    'user': 'myuser',
    'password': 'mypassword'
}

# Step 3: Define Data Processing Logic
def process_sensor_data(sensor_data):
    # Perform data processing and transformation tasks here
    # Example: Inserting the data into a PostgreSQL database
    conn = psycopg2.connect(**db_connection_params)
    cursor = conn.cursor()
    insert_query = "INSERT INTO sensor_data (timestamp, value) VALUES (%s, %s)"
    cursor.execute(insert_query, (sensor_data['timestamp'], sensor_data['value']))
    conn.commit()
    cursor.close()
    conn.close()

# Step 4: Start Data Ingestion and Processing
for message in consumer:
    sensor_data = message.value
    process_sensor_data(sensor_data)

#1.cDevelop a data ingestion pipeline that handles data from different file formats (CSV, JSON, etc.) and performs data validation and cleansing.

import csv
import json
import os

# Step 1: Define Data Validation and Cleansing Functions
def validate_and_cleanse_data(data):
    # Implement data validation and cleansing logic here
    # Example: Remove missing values or perform data type conversions
    cleansed_data = []
    for row in data:
        if all(value.strip() for value in row):  # Check for missing values
            cleansed_row = [value.strip() for value in row]  # Remove leading/trailing whitespaces
            cleansed_data.append(cleansed_row)
    return cleansed_data

# Step 2: Define File Parsing Functions for Different Formats
def parse_csv_file(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

def parse_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Step 3: Define File Ingestion Pipeline
def ingest_data_from_file(file_path):
    _, file_extension = os.path.splitext(file_path)

    if file_extension == '.csv':
        data = parse_csv_file(file_path)
    elif file_extension == '.json':
        data = parse_json_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    cleansed_data = validate_and_cleanse_data(data)
    # Perform additional processing or store the cleansed data as desired
    return cleansed_data

# Usage Example
if __name__ == '__main__':
    file_path = 'data.csv'
    cleansed_data = ingest_data_from_file(file_path)
    print(cleansed_data)

#Q3. Model Validation:
   #a. Implement cross-validation to evaluate the performance of a regression model for predicting housing prices.

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Load the Boston Housing dataset
boston = load_boston()
X, y = boston.data, boston.target

# Create a linear regression model
model = LinearRegression()

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Convert negative mean squared error to positive root mean squared error
rmse_scores = np.sqrt(-scores)

# Print the root mean squared error scores
print("RMSE scores:", rmse_scores)
print("Mean RMSE:", rmse_scores.mean())


#Q3b.Perform model validation using different evaluation metrics such as accuracy, precision, recall, and F1 score for a binary classification problem.
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

# Load the Breast Cancer dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Create a logistic regression model
model = LogisticRegression()

# Perform cross-validation
accuracy_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
precision_scores = cross_val_score(model, X, y, cv=5, scoring='precision')
recall_scores = cross_val_score(model, X, y, cv=5, scoring='recall')
f1_scores = cross_val_score(model, X, y, cv=5, scoring='f1')

# Print the evaluation metrics
print("Accuracy scores:", accuracy_scores)
print("Mean Accuracy:", accuracy_scores.mean())
print("Precision scores:", precision_scores)
print("Mean Precision:", precision_scores.mean())
print("Recall scores:", recall_scores)
print("Mean Recall:", recall_scores.mean())
print("F1 scores:", f1_scores)
print("Mean F1-score:", f1_scores.mean())


#Q3c.Design a model validation strategy that incorporates stratified sampling to handle imbalanced datasets.

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

# Load the Breast Cancer dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Create a logistic regression model
model = LogisticRegression()

# Define the number of folds for stratified sampling
n_folds = 5

# Initialize lists to store evaluation metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Perform stratified sampling and model evaluation
stratified_kfold = StratifiedKFold(n_splits=n_folds)
for train_index, test_index in stratified_kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred))
    recall_scores.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

# Print the evaluation metrics
print("Accuracy scores:", accuracy_scores)
print("Mean Accuracy:", np.mean(accuracy_scores))
print("Precision scores:", precision_scores)
print("Mean Precision:", np.mean(precision_scores))
print("Recall scores:", recall_scores)
print("Mean Recall:", np.mean(recall_scores))
print("F1 scores:", f1_scores)
print("Mean F1-score:", np.mean(f1_scores))


#Q4a.Create a deployment strategy for a machine learning model that provides real-time recommendations based on user interactions.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Model Training
data = pd.read_csv('user_interaction_data.csv')  # Load historical user interaction data
X = data.drop('recommendation', axis=1)
y = data['recommendation']

# Preprocess data (e.g., handle missing values, encode categorical variables)
# ...

# Split data into training and evaluation sets
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a machine learning model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 2: Model Evaluation
y_pred = model.predict(X_eval)
precision = precision_score(y_eval, y_pred)
recall = recall_score(y_eval, y_pred)
print("Precision:", precision)
print("Recall:", recall)

# Step 3: Real-Time Data Collection (Assuming real-time data is received in a pandas DataFrame named 'realtime_data')

# Step 4: Data Preprocessing for Real-Time Recommendations
# Preprocess real-time user interaction data (e.g., handle missing values, encode categorical variables)
# ...

# Step 5: Real-Time Recommendation Generation
realtime_X = realtime_data.drop('recommendation', axis=1)
recommendations = model.predict(realtime_X)

#Q4b.b. Develop a deployment pipeline that automates the process of deploying machine learning models to cloud platforms such as AWS or Azure.
print("doubt in second part")

# 4.c. Design a monitoring and maintenance strategy for deployed models to ensure their performance and reliability over time.

#Data Monitoring:

#Continuously monitor the input data distribution and characteristics to detect any shifts or anomalies.
#Implement data validation checks to ensure the incoming data meets expected criteria.
#Track data quality metrics such as missing values, outliers, or data imbalance.
#Performance Monitoring:

#Monitor the model's performance metrics such as accuracy, precision, recall, or F1-score.
#Set up automated alerts or notifications for significant changes in performance.
#Compare the model's performance against defined thresholds or benchmarks.
#Drift Detection:

#Continuously monitor the model's prediction drift by comparing the model's outputs with ground truth or human experts.
#Track performance metrics specifically for new data or recent time periods to identify potential drift.
#Implement statistical techniques or anomaly detection algorithms to detect significant deviations in model behavior.
#Feedback Loop:
#Establish mechanisms to collect user feedback and integrate it into the model maintenance process.
#Set up channels for users to report issues, provide suggestions, or indicate misclassifications.
#Regularly analyze and incorporate feedback to improve the model's performance and address user concerns.
R#egular Retraining:

#Define a retraining schedule based on the availability of new data and the model's performance trends.
#Retrain the model periodically using updated data to capture evolving patterns and maintain accuracy.
#Incorporate techniques like online learning or incremental training to update the model with new observations.
