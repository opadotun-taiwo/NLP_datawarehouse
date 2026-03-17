import duckdb

con = duckdb.connect("trips.duckdb")

con.execute("""
CREATE TABLE trips AS
SELECT *
FROM read_parquet(
'C:/Users/admin/Desktop/complete_data_project/text_to_sql/yellow_tripdata_2024-01.parquet'
)
""")

print("Data loaded successfully!")