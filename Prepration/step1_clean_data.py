import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


file_data = "data/queenCsvOut.csv"


if not os.path.exists(file_data):
    print("File not found. PLease put the queenCsvOut in the location: data/queenCsvOut.csv")
    exit()
df = pd.read_csv(file_data, skiprows=[1])

'''
plot_df = df[df.LATITUDE>48.9]
plt.scatter(plot_df.LONGITUDE, plot_df.LATITUDE, s=1)
plot_df = df[df.LATITUDE>48.9]
tmp = plot_df[(plot_df["LATITUDE"]<49.1) & (plot_df["LONGITUDE"]> -123.4)]
plt.scatter(tmp.LONGITUDE, tmp.LATITUDE, s=2)
'''

# remove trips that are nether N-H nor H-N trips

print(df[pd.isna(df.LONGITUDE) & (df.THRUST_1==0) &(df.THRUST_2==0)].sum()[df[pd.isna(df.LONGITUDE) & (df.THRUST_1==0) &(df.THRUST_2==0)].sum()==0])

df = df[~(pd.isna(df.LONGITUDE) & (df.THRUST_1==0) &(df.THRUST_2==0))]

# trip ID

# divide the data into trips, each with a unique trip id. The records have trip id 0 when the vessel is parking at bay
# Horseshoe Bay
# 49.3771, -123.2715
H_lat = 49.3771
H_long = -123.2715

# Nanaimo 
# 49.1936, -123.9554
N_lat = 49.1936
N_long = -123.9554

# Give each trip an ID (from H to N or N to H is counted as a complete trip)
# thresh: threshold for the area of the bay
def number_trip(bay_thresh = 1e-6, speed_thres=1):
    trip = np.zeros(df.shape[0])
    trip_id = 1
    trip[0] = trip_id
    prev_at_bay = True
    flag = True
    for i in range(1, len(df)):
        if (i % 50000)==0:
            print(i, len(df))
        H_dist = (df.iloc[i].LONGITUDE - H_long)**2 + (df.iloc[i].LATITUDE - H_lat)**2
        N_dist = (df.iloc[i].LONGITUDE - N_long)**2 + (df.iloc[i].LATITUDE - N_lat)**2
        # decide if the vessel in near the bay
        at_bay = (H_dist < bay_thresh) | (N_dist < bay_thresh)
        if (at_bay):
            # just enter the bay area
            # use flag to check if a new trip is counted
            if (prev_at_bay==False):
                flag = False
            # slows down, means the vessel is likely to be arrived
            # if hasn't generate a new trip id, do so
            if (flag==False) & (df.iloc[i].SOG <= speed_thres):
                trip_id += 1
                flag = True
            # if the vessel speed is very low near the bay area
            # parking at the bay, set trip id to 0
            if (df.iloc[i].SOG <= speed_thres):
                trip[i] = 0
            else:
                trip[i] = trip_id
        else:
            # if leave the bay, but new trip id hasn't been assigned
            # assgin new trip id
            if flag==False:
                trip_id += 1
                flag=True
            trip[i] = trip_id
        prev_at_bay = at_bay
    return trip

df.dropna(axis=0, thresh=35, inplace=True)
df["trip_id"] = number_trip().astype(int)


print(df.trip_id.min(), df.trip_id.max())

print(df.groupby(df.trip_id).count().head(15).Dati)



# These are trips with extremely off locations, drop these trips
off_locations = list(df[df.LATITUDE<49.1].trip_id.unique())
df = df[~df.trip_id.isin(off_locations)]

# plt.scatter(df[df.trip_id==2120].LONGITUDE, df[df.trip_id==2120].LATITUDE, s=1)

# Clean up grouping
# The grouping is not very accurate, for example, there are incomplete trips due to missingdata
#  and abnormal trips that has the same starting and end point. This could be caused by two or 
# more trips classfied into one (long_trips), or a short movement misclassfied into a single trip (short_trips). 

# Typically, a trip from N-H or H-N will take around 100-107 mins (with corresponding lines of data), the median is at 104.
# Apperently any trip too long or too short is questionable.


print(df.groupby(df.trip_id).count().Dati.quantile(0.25), df.groupby(df.trip_id).count().Dati.quantile(0.5), df.groupby(df.trip_id).count().Dati.quantile(0.75))


def get_discontinuous_trip(df, time_thres=10):
    discontinous = {}
    trips = list(df.trip_id.unique())
    trips.remove(0)
    for i in trips:
        tmp_df = df[df.trip_id==i].reset_index()
        fill_in = tmp_df.Time.max()
        tmp_df["time_up"] = tmp_df["Time"].shift(periods=-1, fill_value=fill_in)
        tmp_df["time_diff"] = tmp_df["time_up"] - tmp_df["Time"]
        if tmp_df.time_diff.max() > time_thres:
            discontinous[i] = tmp_df.time_diff.max() 
    return discontinous

trip_dict = get_discontinuous_trip(df, time_thres=10)
trips_to_process = trip_dict.keys()
# trip_dict


#For the discontinuous trips, we only preserve the longer (more complete) parts, delete the shorter parts.

def find_drop_times(df, long_trips):
    to_drop = []
    for trip_id in long_trips:
        times = list(df[df.trip_id==trip_id].Time)
        count = 0
        i = 0
        # the jump in time is less than 10 mins
        while((i<len(times)-2) and ((times[i+1]-times[i])< 10)):
            count = count+1
            i = i+1
        # if the more complete trip happens earlier
        if count < len(times)/2:
            drop_time = times[0:count+1]
        elif count+1 == len(times)-1:
            drop_time = [times[-1]]
        else:
            drop_time = times[count+1:-1]
        to_drop = to_drop + drop_time
    return to_drop

# drop the shorter parts in the discontinuous trips
time_to_drop = find_drop_times(df, trip_dict)
df = df[~df.Time.isin(time_to_drop)]
print(get_discontinuous_trip(df, time_thres=10).keys())


trips_to_process = get_discontinuous_trip(df, time_thres=10).keys()
time_to_drop = find_drop_times(df, trip_dict)
df = df[~df.Time.isin(time_to_drop)]
print(get_discontinuous_trip(df, time_thres=10))

# plt.scatter(df[df.trip_id==3075].LONGITUDE, df[df.trip_id==3075].LATITUDE, s=2, c=df[df.trip_id==3075].Time, cmap="BrBG")
df = df[[df.trip_id!= i for i in get_discontinuous_trip(df, time_thres=10).keys()]]

#After slicing, we will have trips that starts or ends too far away from the bay.\
# In other words, they are incomplete trips. Remove these incomplete trips


def incomplete_trips(df, thresh=1e-3):
    incomplete_trips = []
    trips = list(df.trip_id.unique())
    trips.remove(0)
    for trip_id in trips:
        tmp_df = df[df.trip_id==trip_id].reset_index()
        start_H_dist = (tmp_df.iloc[0].LONGITUDE - H_long)**2 + (tmp_df.iloc[0].LATITUDE - H_lat)**2
        start_N_dist = (tmp_df.iloc[0].LONGITUDE - N_long)**2 + (tmp_df.iloc[0].LATITUDE - N_lat)**2
        start_bay_dist = min(start_H_dist, start_N_dist)
        end_H_dist = (tmp_df.iloc[-1].LONGITUDE - H_long)**2 + (tmp_df.iloc[-1].LATITUDE - H_lat)**2
        end_N_dist = (tmp_df.iloc[-1].LONGITUDE - N_long)**2 + (tmp_df.iloc[-1].LATITUDE - N_lat)**2
        end_bay_dist = min(end_H_dist, end_N_dist)
        travel_dist = (tmp_df.iloc[0].LONGITUDE - tmp_df.iloc[-1].LONGITUDE)**2 + \
            (tmp_df.iloc[0].LATITUDE - tmp_df.iloc[-1].LATITUDE)**2
        if (start_bay_dist>thresh) | (end_bay_dist>thresh) | (travel_dist<.15):
            incomplete_trips.append(trip_id)
    df = df[~df.trip_id.isin(incomplete_trips)]
    print(incomplete_trips)
    return df
df = incomplete_trips(df, 1e-3)
df.groupby(df.trip_id).count().Dati[df.groupby(df.trip_id).count().Dati<60]


print(df.groupby(df.trip_id).count().Dati[df.groupby(df.trip_id).count().Dati<90])

# plt.scatter(df[df.trip_id==3111].LONGITUDE, df[df.trip_id==3111].LATITUDE, s=2, c=df[df.trip_id==3111].Time, cmap="BrBG")
# plt.scatter(df[df.trip_id==3481].LONGITUDE, df[df.trip_id==3481].LATITUDE, s=2, c=df[df.trip_id==3481].Time, cmap="BrBG")

# trip 3111 and 3481 are another two special cases, they do not have any jump in time, \
# in other words, no missing data in between, but they have a location jump\
# remove them from the data set

df = df[~df.trip_id.isin([3111,3481])]
print(df.groupby(df.trip_id).count().Dati[df.groupby(df.trip_id).count().Dati>150])

plt.scatter(df[df.trip_id==3733].LONGITUDE, df[df.trip_id==3733].LATITUDE, s=2, c=df[df.trip_id==3733].Time, cmap="BrBG")


# 3733 is a special case where a detour happened

print(df.groupby(df.trip_id).count().Dati.min(), df[df.trip_id!=0].groupby(df.trip_id).count().Dati.max())


#Another type of abnormal trips are those that has the same starting and end point. \
# This could be caused by two or more trips classfied into one (long_trips), or a short movement misclassfied into a single trip (short_trips). \
# We nolonger have these type of trips after previous steps to clean up the data.

def find_round_trip(df):
    short_trip = []
    long_trip = []
    trips = list(df.trip_id.unique())
    trips.remove(0)
    for i in trips:
        tmp_df = df[df.trip_id == i].reset_index()
        start_long, start_lat = tmp_df.iloc[0].LONGITUDE,tmp_df.iloc[0].LATITUDE
        end_long, end_lat = tmp_df.iloc[-1].LONGITUDE,tmp_df.iloc[-1].LATITUDE
        if (abs(start_long - end_long) < 0.005) and (abs(start_lat- end_lat) < 0.005):
            min_long, min_lat = tmp_df.LONGITUDE.min(), tmp_df.LATITUDE.min()
            max_long, max_lat = tmp_df.LONGITUDE.max(), tmp_df.LATITUDE.max()
            if ((max_long-min_long)>0.1) | ((max_lat-min_lat)>0.1):
                long_trip.append(i)
            else:
                short_trip.append(i)
    return short_trip, long_trip
print(find_round_trip(df))

df.trip_id = df.trip_id.astype(int)

print(df.groupby("trip_id").count().Dati.quantile(.01), df.groupby("trip_id").count().Dati.median(), df.groupby("trip_id").count().Dati.quantile(.99))
print(df.groupby(df.trip_id).count().Dati.min(), df[df.trip_id!=0].groupby(df.trip_id).count().Dati.max())

plt.scatter(df.LONGITUDE, df.LATITUDE, s=1)

df.to_csv("data/queenCsvOut_cleaned_location.csv", index=False)