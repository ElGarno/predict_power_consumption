# forecast_next_day.py

# 1) pip install wetterdienst
import pandas as pd
from datetime import datetime, timedelta
from wetterdienst.provider.dwd.mosmix import DwdMosmixRequest, DwdMosmixType
from wetterdienst import Settings

def fetch_tomorrow_forecast(station_ids: list[str]) -> pd.DataFrame:
    settings = Settings(ts_shape="long", ts_humanize=False)
    request = DwdMosmixRequest(
        parameter=["TTT", "RR1c", "SunD1", "FF", "N"],
        mosmix_type=DwdMosmixType.SMALL,
        settings=settings
    )
    stations = request.filter_by_station_id(station_id=station_ids)
    df = stations.values.all().df

    df = (
        df
        .pivot_table(index=["station_id", "date"], columns="parameter", values="value")
        .reset_index()
        .rename(columns={
            "TTT": "temperature",
            "RR1c": "precipitation",
            "SunD1": "sunshine_duration",
            "FF": "wind_speed",
            "N": "cloud_cover"
        })
    )

    # Convert units
    df["temperature"] -= 273.15
    df["sunshine_duration"] /= 3600

    tomorrow = (datetime.utcnow() + timedelta(days=1)).date()
    return df[df["date"].dt.date == tomorrow]

if __name__ == "__main__":
    station_ids = ["02947", "00216", "01639", "03098", "03098"]
    forecast_df = fetch_tomorrow_forecast(station_ids)
    print(forecast_df.sort_values(["station_id", "date"]))