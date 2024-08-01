# sql_operations.py

import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user='root',
        password='password',
        database='new_schema'
    )

def insert_predictions(predictions):
    conn = get_connection()
    cursor = conn.cursor()

    insert_query = """
    INSERT INTO predictions (ds, start_time_yhat1, end_time_yhat1, tripmiles_yhat1)
    VALUES (%s, %s, %s, %s)
    """

    for index, row in predictions.iterrows():
        cursor.execute(insert_query, (row['ds'], row['start_time_yhat1'], row['end_time_yhat1'], row['tripmiles_yhat1']))

    conn.commit()
    cursor.close()
    conn.close()
