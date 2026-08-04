import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus


password = quote_plus("Sriharsha@0704")


engine = create_engine(
    f"mysql+mysqlconnector://root:{password}@localhost/superstore_db"
)


df = pd.read_sql(
    "SELECT * FROM sales_data LIMIT 5",
    engine
)


print(df)