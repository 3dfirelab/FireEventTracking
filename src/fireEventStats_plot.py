import matplotlib.pyplot as plt
import pandas as pd
import pickle

dirIn = '/data/shared/FCI/MED_fire_events/Stats/'
df = pd.read_csv(f'{dirIn}/MED-weekly.csv')
with open(f"{dirIn}/MED-area_duration_weekly_allData.pkl", "rb") as f:
    data_per_week_all = pickle.load(f)

fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

ax=axes[0]
ax.plot(df.week, df.total_frp)
ax.set_ylabel('FRP (MW)')

ax=axes[1]
ax.plot(df.week, df.event_count)
ax.set_ylabel('event count')

dataA = [x[0] for x in data_per_week_all] 
dataT = [x[1] for x in data_per_week_all] 
dataAc = [[x          for i,(x,t) in enumerate(zip(subA,subT)) if (x >= 10) & (t>0)] for subA,subT in zip(dataA,dataT)] # remove 0
dataTc = [[subT[i]/24 for i,(x,t) in enumerate(zip(subA,subT)) if (x >= 10) & (t>0)] for subA,subT in zip(dataA,dataT)] # remove 0


ax=axes[2]
y = df["mean_area"]
y_std = df["std_area"]
#ax.plot(df.week, y)
ax.boxplot(dataAc, positions=df.week,  widths=0.4 )
# ±1σ envelope
#ax.fill_between(
#    df.week,
#    y - y_std,
#    y + y_std,
#    alpha=0.3,
#    label="±1 std. dev.")
ax.set_ylabel('event area ($>10$) [ha]')
ax.set_yscale('log')

ax=axes[3]
#y_std = df["std_duration_h"]
#ax.plot(df.week, y)
ax.boxplot(dataTc, positions=df.week,  widths=0.4 )
# ±1σ envelope
#ax.fill_between(
#    df.week,
#    y - y_std,
#    y + y_std,
#    alpha=0.3,
#    label="±1 std. dev."
#)
ax.set_ylabel('event duratuion ($>0$) [day]')
ax.set_xlabel('week of the year 2025')
ax.set_yscale('log')


fig.suptitle("Weeky Fire Behavior Descripot - MED", fontsize=14)

plt.show()



