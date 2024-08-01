import pandas as pd
from datetime import datetime, timedelta
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data = data[['PERSONID', 'TRIPID', 'TRAVDAY', 'TDAYDATE', 'STRTTIME', 'ENDTIME', 'TRPMILES', 'TRIPPURP', 'WHYFROM', 'WHYTO', 'TRPTRANS']]
    data.sort_values(by='TDAYDATE', inplace=True)
    data = data[(data['PERSONID'] == 1) & (data['WHYFROM'] == 1)]

    data[['STRTTIME_OLD', 'ENDTIME_OLD']] = data[['STRTTIME', 'ENDTIME']]
    data['TRAVDAY'] = data['TRAVDAY'].astype(int)
    data['TRIPPURP'] = data['TRIPPURP'].astype(int)

    def convert_to_time_format(time_int):
        time_str = str(time_int).zfill(4)
        return time_str[:2] + ':' + time_str[2:]

    data['TDAYDATE'] = pd.to_datetime(data['TDAYDATE'].astype(str), format='%Y%m').dt.strftime('%Y-%m')
    data['STRTTIME'] = data['STRTTIME'].apply(convert_to_time_format)
    data['ENDTIME'] = data['ENDTIME'].apply(convert_to_time_format)
    
    data['TDAYDATE'] = data['TDAYDATE'].str.strip()
    data['STRTTIME'] = data['STRTTIME'].str.strip()
    data['ENDTIME'] = data['ENDTIME'].str.strip()

    data['STRTTIME'] = pd.to_datetime(data['TDAYDATE'] + ' ' + data['STRTTIME'], format='%Y-%m %H:%M', errors='coerce')
    data['ENDTIME'] = pd.to_datetime(data['TDAYDATE'] + ' ' + data['ENDTIME'], format='%Y-%m %H:%M', errors='coerce')

    def convert_to_full_date(row):
        first_day_of_month = datetime.strptime(row['TDAYDATE'], '%Y-%m')
        weekday = (row['TRAVDAY'] - 1) % 7
        first_weekday_of_month = first_day_of_month.weekday()
        delta_days = (weekday - first_weekday_of_month + 7) % 7
        full_date = first_day_of_month + timedelta(days=delta_days)
        return full_date

    data['FULLDATE'] = data.apply(convert_to_full_date, axis=1)

    def combine_date_time(row, time_col):
        if pd.isna(row[time_col]):
            return pd.NaT
        datetime_str = f"{row['FULLDATE'].strftime('%Y-%m-%d')} {row[time_col].strftime('%H:%M')}"
        return pd.to_datetime(datetime_str, format='%Y-%m-%d %H:%M')

    data['FULL_DATETIME_STRT'] = data.apply(lambda row: combine_date_time(row, 'STRTTIME'), axis=1)
    data['FULL_DATETIME_END'] = data.apply(lambda row: combine_date_time(row, 'ENDTIME'), axis=1)
    
    
    return data

def prepare_data(df, datetime_column, value_column):
    df_prepared = df[[datetime_column, value_column]].copy()
    df_prepared.rename(columns={datetime_column: 'ds', value_column: 'y'}, inplace=True)
    
    return df_prepared

def train_neuralprophet(df_prepared, epochs=100):
    model = NeuralProphet(
    epochs=50,
    learning_rate=1.0,
    batch_size=64
    )
    
    model.fit(df_prepared, freq='h')
    forecast = model.predict(df_prepared)
    return model, forecast

def forecast_next_day(model, df_prepared, periods=10):
    future = model.make_future_dataframe(df_prepared, periods=periods)
    forecast = model.predict(future)
    return forecast

def convert_yhat1_to_time(yhat1):
    total_minutes = int(yhat1)
    hours = total_minutes // 60
    minutes = total_minutes % 60
    hours = hours % 24  # Ensure hours are within 0-23
    return f"{str(hours).zfill(2)}:{str(minutes).zfill(2)}"

def train_test_split(df, train_size=0.8):
    train_size = int(len(df) * train_size)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    return train_df, test_df

def evaluate_model(test_df, forecast_df):
    y_true = test_df['y'].values
    y_pred = forecast_df['yhat1'].values[:len(y_true)]
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    return mae, mse, rmse

if __name__ == "__main__":
    
    # Step 1: Preprocess Data
    data = preprocess_data('tripv2pub 5.csv')

    

    # End Time Forecast
    data_aggregated_end = data.groupby('FULL_DATETIME_END').agg({
        'ENDTIME_OLD': 'first'
    }).reset_index()
    
    df_endtime = prepare_data(data_aggregated_end, 'FULL_DATETIME_END', 'ENDTIME_OLD')
    

    # Train-test split for end time
    train_df_endtime, test_df_endtime = train_test_split(df_endtime)

    # Train NeuralProphet model for end time
    model_endtime, _ = train_neuralprophet(train_df_endtime)

    # Forecast on the test set for end time
    forecast_endtime = model_endtime.predict(test_df_endtime)

    # Evaluate the model for end time
    mae_endtime, mse_endtime, rmse_endtime = evaluate_model(test_df_endtime, forecast_endtime)
    print(f'End Time Forecast - MAE: {mae_endtime}, MSE: {mse_endtime}, RMSE: {rmse_endtime}')

    # Forecast next day for end time
    forecast_next_day_endtime = forecast_next_day(model_endtime, train_df_endtime)
    forecast_next_day_endtime['end_time_yhat1'] = forecast_next_day_endtime['yhat1'].apply(convert_yhat1_to_time)
    print('Next day End Time forecast:', forecast_next_day_endtime[['ds', 'end_time_yhat1']])

