
from datetime import timedelta
import fastf1
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

fastf1.Cache.enable_cache("f1_cache")

race = fastf1.get_session(2025, "British Grand Prix", "R")
race.load()

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)


session_info = race.session_info
start_date = session_info['StartDate'].strftime("%Y-%m-%d")
end_date = session_info['EndDate']

total_laps = race.total_laps

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(total_laps)

#print(start_date)
#print(end_date)

