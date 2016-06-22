import pandas as pd
import numpy as np
from geopy.distance import vincenty, great_circle
from sklearn.cross_validation import train_test_split, cross_val_score
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import matplotlib.pyplot as plt
import itertools
from sklearn.ensemble import RandomForestRegressor as RF


def haversine_np(df, coordinates):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    lng1, lat1, lng2, lat2 = map(np.radians, [df[coordinate] for coordinate in coordinates])

    dlon = lng2 - lng1
    dlat = lat2 - lat1

    sin_lat1, cos_lat1 = np.sin(lat1), np.cos(lat1)
    sin_lat2, cos_lat2 = np.sin(lat2), np.cos(lat2)

    delta_lng = lng2 - lng1
    cos_delta_lng, sin_delta_lng = np.cos(delta_lng), np.sin(delta_lng)

    d = np.arctan2(np.sqrt((cos_lat2 * sin_delta_lng) ** 2 + \
        (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_delta_lng) ** 2), \
        sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng)

    mile = 3959 * d
    return mile


def plot_trips(df, coordinates):

    trips = np.array(df.reset_index().loc[:200, coordinates])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for trip in trips:
        plt.plot([trip[0], trip[2]], [trip[1], trip[3]], color='brown', marker='o')
    plt.show()


def feature_engineer(df, coordinates):

    df['distance'] = haversine_np(df, coordinates)
    df['start_timestamp_new'] = pd.to_datetime(df['start_timestamp'], unit='s')
    df['DOW'] = pd.DatetimeIndex(df['start_timestamp_new'], unit='s').dayofweek
    df['hour'] = pd.DatetimeIndex(df['start_timestamp_new'], unit='s').hour

    cal = calendar()
    holidays = cal.holidays(start=df['start_timestamp_new'].min(), end=df['start_timestamp_new'].max())
    df['holiday'] = df['start_timestamp_new'].isin(holidays)


if __name__ == "__main__":
    train = pd.read_csv("../travel_time/train.csv")
    test = pd.read_csv("../travel_time/test.csv")

    coordinates = ['start_lng', 'start_lat', 'end_lng', 'end_lat']
    feature_engineer(train, coordinates)
    feature_engineer(test, coordinates)

    features = ['distance', 'DOW', 'hour']



    training, validation = train_test_split(train, test_size=0.1, random_state=42)
    # plot_trips(validation, coordinates)

    x_train = validation[features]
    y_train = validation['duration']

    score = cross_val_score(RF(), x_train, y_train, cv=3, n_jobs=-1, scoring='mean_squared_error').mean()
