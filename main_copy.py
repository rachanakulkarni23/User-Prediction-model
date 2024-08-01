# main.py

from neural_prophet_strt import preprocess_data, prepare_data, train_neuralprophet, forecast_next_day,convert_yhat1_to_time
from sql_operations import insert_predictions
from optimize import optimize
from db_connections import fetch_data
import subprocess
def main():
    data = preprocess_data('tripv2pub 5.csv')

    # Start Time Forecast
    data_aggregated_start = data.groupby('FULL_DATETIME_STRT').agg({
        'STRTTIME_OLD': 'first',
        'ENDTIME_OLD': 'first',
        'TRPMILES': 'first'
    }).reset_index()

    df_strttime = prepare_data(data_aggregated_start, 'FULL_DATETIME_STRT', 'STRTTIME_OLD')
    model_strttime, forecast_strttime = train_neuralprophet(df_strttime)
    forecast_next_day_strttime = forecast_next_day(model_strttime, df_strttime)
    forecast_next_day_strttime['start_time_yhat1'] = forecast_next_day_strttime['yhat1'].apply(convert_yhat1_to_time)

    # # End Time Forecast
    # data_aggregated_end = data.groupby('FULL_DATETIME_END').agg({
    #     'ENDTIME_OLD': 'first'
    # }).reset_index()

    df_endtime = prepare_data(data_aggregated_start, 'FULL_DATETIME_STRT', 'ENDTIME_OLD')
    model_endtime, forecast_endtime = train_neuralprophet(df_endtime)
    forecast_next_day_endtime = forecast_next_day(model_endtime, df_endtime)
    forecast_next_day_endtime['end_time_yhat1'] = forecast_endtime['yhat1'].apply(convert_yhat1_to_time)

    df_trpmiles = prepare_data(data_aggregated_start, 'FULL_DATETIME_STRT', 'TRPMILES')
    model_trpmiles, forecast_trpmiles = train_neuralprophet(df_trpmiles)
    forecast_next_day_trpmiles = forecast_next_day(model_trpmiles, df_trpmiles)
    forecast_next_day_trpmiles['trpmiles_yhat1'] = forecast_trpmiles['yhat1']

    # Merge start and end time forecasts
    # predictions = forecast_next_day_strttime[['ds', 'start_time_yhat1']].copy()
    # predictions['end_time_yhat1'] = forecast_next_day_endtime['yhat1'].apply(convert_yhat1_to_time)
    # predictions['trpmiles_yhat1'] = forecast_next_day_trpmiles['yhat1']

    # Save predictions to CSV and insert into database
    forecast_next_day_trpmiles.to_csv('predictions.csv', index=False)
    insert_predictions(forecast_next_day_trpmiles)

    # Fetch data and optimize
    data = fetch_data()
    optimize(data)
    # subprocess.run(['python3', 'optimize.py'])
if __name__ == "__main__":
    main()
