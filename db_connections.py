import mysql.connector

def fetch_data():
    # Connect to the MySQL database
    connection = mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user='root',
        password='password',
        database='new_schema'
    )

    cursor = connection.cursor()

    # Execute a query to fetch data
    query = "SELECT * FROM predictions"
    cursor.execute(query)

    # Fetch all the rows from the executed query
    rows = cursor.fetchall()

    # Close the cursor and connection
    cursor.close()
    connection.close()

    return rows
