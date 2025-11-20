
from datetime import timedelta
import fastf1
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

fastf1.Cache.enable_cache("f1_cache")

race = fastf1.get_session(2021, "Belgium", "R")
race.load()

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)


session_info = race.session_info
start_date = session_info['StartDate'].strftime("%Y-%m-%d")
end_date = session_info['EndDate']

print(start_date)
print(end_date)

