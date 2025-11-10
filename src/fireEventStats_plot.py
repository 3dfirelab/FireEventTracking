import matplotlib.pyplot as plt
import pandas as pd

dirIn = '/data/shared/FCI/MED_fire_events/Fig/'
df = pd.read_csv(f'{dirIn}/MED-weekly.csv')


fig=plt.figure(figsize=(12,12))

ax=plt.subplot(411)
ax.plot(df.week, df.total_frp)
ax.set_ylabel('FRP (MW)')

ax=plt.subplot(412)
ax.plot(df.week, df.event_count)
ax.set_ylabel('event count')

ax=plt.subplot(413)
y = df["mean_area"]
y_std = df["std_area"]
ax.plot(df.week, y)
# ±1σ envelope
ax.fill_between(
    df.week,
    y - y_std,
    y + y_std,
    alpha=0.3,
    label="±1 std. dev.")
ax.set_ylabel('mean event area [ha]')

ax=plt.subplot(414)
y = df["mean_duration_h"]
y_std = df["std_duration_h"]
ax.plot(df.week, y)
# ±1σ envelope
ax.fill_between(
    df.week,
    y - y_std,
    y + y_std,
    alpha=0.3,
    label="±1 std. dev."
)
ax.set_ylabel('mean event duratuion [h]')
ax.set_xlabel('week of the year 2025')


fig.suptitle("Weeky Fire Behavior Descripot - MED", fontsize=14)

plt.show()



