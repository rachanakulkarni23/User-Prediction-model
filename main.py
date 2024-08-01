# main.py

from neural_prophet_strt import preprocess_data as preprocess_strt, prepare_data as prepare_data_strt, train_neuralprophet as train_neuralprophet_strt, forecast_next_day as forecast_next_day_strt, convert_yhat1_to_time
from neural_prophet_end import preprocess_data as preprocess_end, prepare_data as prepare_data_end, train_neuralprophet as train_neuralprophet_end, forecast_next_day as forecast_next_day_end, convert_yhat1_to_time
from neural_prophet_tripmiles import preprocess_data as preprocess_tripmiles, prepare_data as prepare_data_tripmiles, train_neuralprophet as train_neuralprophet_tripmiles, forecast_next_day as forecast_next_day_tripmiles
from sql_operations import insert_predictions
from db_connections import fetch_data
from optimize import optimize
import pandas as pd

def main():
    # Step 1: Preprocess Data
    data_strt = preprocess_strt('tripv2pub 5.csv')
    data_tripmiles, scaler_tripmiles = preprocess_tripmiles('tripv2pub 5.csv')

    # Step 2: Start Time Forecast
    data_aggregated_strt = data_strt.groupby('FULL_DATETIME_STRT').agg({
        'STRTTIME_OLD': 'first'
    }).reset_index()

    df_strttime = prepare_data_strt(data_aggregated_strt, 'FULL_DATETIME_STRT', 'STRTTIME_OLD')
    model_strttime, forecast_strttime = train_neuralprophet_strt(df_strttime)
    forecast_next_day_strttime = forecast_next_day_strt(model_strttime, df_strttime)
    forecast_next_day_strttime['start_time_yhat1'] = forecast_next_day_strttime['yhat1'].apply(convert_yhat1_to_time)

    # Step 3: End Time Forecast
    data_aggregated_end = data_strt.groupby('FULL_DATETIME_END').agg({
        'ENDTIME_OLD': 'first'
    }).reset_index()

    df_endtime = prepare_data_end(data_aggregated_end, 'FULL_DATETIME_END', 'ENDTIME_OLD')
    model_endtime, forecast_endtime = train_neuralprophet_end(df_endtime)
    forecast_next_day_endtime = forecast_next_day_end(model_endtime, df_endtime)
    forecast_next_day_endtime['end_time_yhat1'] = forecast_next_day_endtime['yhat1'].apply(convert_yhat1_to_time)

    # Step 4: Trip Miles Forecast
    data_aggregated_tripmiles = data_tripmiles.groupby('FULL_DATETIME_STRT').agg({
        'TRPMILES': 'mean'
    }).reset_index()

    df_trpmiles = prepare_data_tripmiles(data_aggregated_tripmiles, 'FULL_DATETIME_STRT', 'TRPMILES')
    model_trpmiles, forecast_trpmiles = train_neuralprophet_tripmiles(df_trpmiles)
    forecast_next_day_trpmiles = forecast_next_day_tripmiles(model_trpmiles, df_trpmiles)
    forecast_next_day_trpmiles['tripmiles_yhat1'] = scaler_tripmiles.inverse_transform(forecast_next_day_trpmiles[['yhat1']])

    # Combine Forecasts
    predictions = forecast_next_day_strttime[['ds', 'start_time_yhat1']].copy()
    predictions['end_time_yhat1'] = forecast_next_day_endtime['end_time_yhat1']
    predictions['tripmiles_yhat1'] = forecast_next_day_trpmiles[['tripmiles_yhat1']]


    predictions.to_csv('predictions.csv', index=False)

    # Step 5: Insert Predictions into Database
    insert_predictions(predictions)

    # Fetch data and optimize
    data = fetch_data()
    optimize(data)

if __name__ == "__main__":
    main()
