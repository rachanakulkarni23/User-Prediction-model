# main.py

from neural_prophet import preprocess_data, prepare_data, train_neuralprophet, forecast_next_day
from sql_operations import insert_predictions

def main():
    # Load and preprocess data
    data = preprocess_data('tripv2pub 5.csv')
    
    # Prepare data for modeling
    data_aggregated = data.groupby('FULL_DATETIME_STRT').agg({
        'STRTTIME_OLD': 'first'
    }).reset_index()

    df_strttime = prepare_data(data_aggregated, 'FULL_DATETIME_STRT', 'STRTTIME_OLD')
    
    # Train model and make predictions
    model_strttime, forecast_strttime = train_neuralprophet(df_strttime)
    forecast_next_day_strttime = forecast_next_day(model_strttime, df_strttime)

    # Save predictions to CSV and insert into database
    forecast_next_day_strttime.to_csv('predictions.csv', index=False)
    insert_predictions(forecast_next_day_strttime)

if __name__ == "__main__":
    main()
