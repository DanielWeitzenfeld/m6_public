import yaml
import requests

import pandas as pd

import m6_public


class EOD:
    endpoint = "https://eodhistoricaldata.com/api"

    def __init__(self):
        self.config = self.read_configuration()
        self.last_request = None

    def read_configuration(self):
        with open(m6_public.ROOT + "/etc/eod.yml") as file:
            return yaml.load(file, Loader=yaml.FullLoader)

    def get_data_for_ticker(
        self,
        ticker,
        exchange,
        start_date=None,
        end_date=None,
    ):
        url = self.endpoint + "/eod/" + f"{ticker}.{exchange}"
        params = {"api_token": self.config["key"], "fmt": "json"}
        if start_date:
            params["from"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["to"] = end_date.strftime("%Y-%m-%d")
        r = requests.get(url, params=params)
        df = pd.DataFrame(r.json())
        df.date = pd.to_datetime(df.date)
        return df

    def get_earnings_for_ticker(
        self,
        ticker,
        exchange,
        start_date=None,
        end_date=None,
    ):
        url = self.endpoint + "/calendar/earnings"
        params = {
            "api_token": self.config["key"],
            "fmt": "json",
            "symbols": f"{ticker}.{exchange}",
        }
        if start_date:
            params["from"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["to"] = end_date.strftime("%Y-%m-%d")
        r = requests.get(url, params=params)
        self.last_request = r
        df = pd.DataFrame(r.json())
        df["report_date"] = df.earnings.apply(lambda x: x["report_date"])
        df["ticker"] = ticker
        df.report_date = pd.to_datetime(df.report_date)
        return df
