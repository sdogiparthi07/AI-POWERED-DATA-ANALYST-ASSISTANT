import mysql.connector
# Connect to MySQL server
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Sriharsha@0704"
)

cursor = connection.cursor()


# Create database
cursor.execute(
    "CREATE DATABASE IF NOT EXISTS superstore_db"
)

print("Database created successfully")


cursor.close()
connection.close()


import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus


# Load cleaned dataset
df = pd.read_csv(
    "data/cleaned_superstore.csv"
)


# MySQL password
password = quote_plus("Sriharsha@0704")


# Create connection
engine = create_engine(
    f"mysql+mysqlconnector://root:{password}@localhost/superstore_db"
)


# Upload data
df.to_sql(
    name="sales_data",
    con=engine,
    if_exists="replace",
    index=False
)


print("Data uploaded successfully to MySQL")

