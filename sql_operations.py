# database.py

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
    
    insert_query = "INSERT INTO new_table (ds, y_pred) VALUES (%s, %s)"
    
    for index, row in predictions.iterrows():
        cursor.execute(insert_query, (row['ds'], row['yhat1']))
    
    conn.commit()
    cursor.close()
    conn.close()
