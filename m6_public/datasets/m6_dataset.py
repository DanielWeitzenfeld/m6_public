from datetime import datetime
import os
import pandas as pd
import numpy as np

import m6_public
from m6_public.apis import eod as eod_api

eod = eod_api.EOD()


class Dataset:
    # hard code start dates for a few assets
    start_dates = {
        "IUMO.L": datetime(2018, 6, 1),
        "SEGA.L": datetime(2012, 3, 1),
        "CHTR": datetime(2010, 1, 5),
        "IEAA.L": datetime(2017, 10, 25),
    }

    def __init__(
        self,
        min_date=datetime(
            2005,
            1,
            1,
        ),
        anchor_date=datetime(2022, 2, 20),
        logger=None,
    ):
        self.logger = logger
        self.df = pd.read_csv(os.path.join(m6_public.ROOT, "data", "M6_Universe.csv"))
        df = pd.read_csv(os.path.join(m6_public.ROOT, "data", "assets_m6.csv"))
        df.date = pd.to_datetime(df.date)
        df = df.rename(columns={"price": "close"})
        self.df_prices = df
        self.min_date = min_date
        self.anchor_date = anchor_date
        self.df_cal = self.construct_calendar(min_date, anchor_date)
        self.g_cal = self.construct_period_level_calendar(self.df_cal)

    def log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def earnings_for_ticker(self, ticker):
        exchange = self.exchange_for_symbol(ticker)
        ticker_ = "EG" if ticker == "RE" else ticker
        df = eod.get_earnings_for_ticker(
            ticker=ticker_,
            exchange=exchange,
            start_date=self.min_date,
            end_date=datetime(2025, 1, 1),
        )
        return df

    @staticmethod
    def exchange_for_symbol(symbol):
        if "." in symbol:
            exchange = "LSE"
        else:
            exchange = "US"
        return exchange

    def fetch_history(self, symbol, df_cal):
        if "." in symbol:
            symbol_ = symbol.split(".")[0]
            exchange = "LSE"
        else:
            symbol_ = symbol
            exchange = "US"
        symbol_ = "EG" if symbol_ == "RE" else symbol_
        df = eod.get_data_for_ticker(symbol_, exchange).set_index("date")
        last_day_of_data = df.index[df.adjusted_close.notnull()].max()
        self.log("{0}: last_day of data: {1}".format(symbol, last_day_of_data))

        if symbol in self.start_dates:
            df = df[df.index >= self.start_dates[symbol]]

        df = df.drop("close", axis=1).rename(columns={"adjusted_close": "close"})

        # reindex and ffill
        new_index = pd.date_range(df.index.min(), df.index.max(), freq="B")
        df = df.reindex(new_index)
        df.close = df.close.ffill()
        # infill prices with those provided by m6
        df_m6 = self.df_prices[self.df_prices.symbol == symbol].copy()
        df_m6 = df_m6.set_index("date")
        needed_dates = set(df_m6.index.values) - set(df.index.values)
        if needed_dates:
            self.log("{0} needed dates for {1}".format(len(needed_dates), symbol))
            df_m6 = df_m6.loc[needed_dates]
            df_m6 = df_m6[["close"]].rename(columns={"close": "close_m6"})
            df = df.join(df_m6, how="outer")
            df.close = df.close.fillna(df.close_m6)
            df = df.sort_index()
            df.open = df.open.fillna(df.close.shift(1))

        if symbol in self.start_dates:
            df = df[df.index >= self.start_dates[symbol]]
        # Feature engineering.  Note that not all features are used in the model
        df["d1_r"] = (df.close - df.close.shift(1)) / df.close.shift(1)
        min_date_with_data = df.index.min()
        df = pd.merge(df, df_cal, left_index=True, right_on="date", how="right")
        df = df[df.date >= min_date_with_data]
        drop_us = []
        min_period = df.period_i.min()
        if df[df.period_i == min_period].shape[0] < 20:
            drop_us.append(min_period)
            if df[(df.period_i == min_period) & (df.d1_r.notnull())].shape[0] < 5:
                drop_us.append(min_period + 1)
        g = df.groupby("period_i")
        g = pd.DataFrame(
            {
                "d20_std": g.d1_r.std(),
                "close": g.close.last(),
            }
        )
        g_last_week = df[df.is_last_week].groupby("period_i")
        g_last_week = pd.DataFrame(
            {
                "d5_std": g_last_week.d1_r.std(),
                "open_5days_left": g_last_week.open.first(),
            }
        )
        g = g.join(g_last_week)

        g_last_fortnight = df[df.is_last_fortnight].groupby("period_i")
        g_last_fortnight = pd.DataFrame(
            {
                "d10_std": g_last_fortnight.d1_r.std(),
                "open_10days_left": g_last_fortnight.open.first(),
            }
        )

        g = g.join(g_last_fortnight)

        g["d20_r"] = (g.close - g.close.shift(1)) / g.close.shift(1)
        g["d5_r"] = (g.close - g.open_5days_left) / g.open_5days_left
        g["d10_r"] = (g.close - g.open_10days_left) / g.open_10days_left
        g["d5_r_lag1"] = g.d5_r.shift(1)
        g["d5_std_lag1"] = g.d5_std.shift(1)
        g["d10_r_lag1"] = g.d10_r.shift(1)

        if drop_us:
            g = g[~g.index.isin(drop_us)]
        g = g[g.index >= 0]
        g = g[g.d5_r_lag1.notnull()]

        if symbol == "DRE":
            # this is when DRE was delisted, assuming self.min_date is 1/1/2005.
            g = g[g.index < 230]

        return g

    def construct_calendar(self, min_date, anchor_date):
        forecast_period = anchor_date + pd.tseries.offsets.Week(4)
        wk4_dates = pd.date_range(end=forecast_period, freq="4W", periods=252)
        wk4_dates = [d for d in wk4_dates if d >= min_date]
        every_day = pd.date_range(np.min(wk4_dates), forecast_period, freq="D")
        df_cal = pd.DataFrame({"date": every_day})
        df_cal.loc[df_cal.date.isin(wk4_dates), "period_i"] = 1
        df_cal.period_i = df_cal.period_i.fillna(0).cumsum() - 2
        df_cal = df_cal[df_cal.date.dt.weekday <= 4]
        df_cal["day_within_period"] = df_cal.groupby("period_i").cumcount()
        df_cal["is_last_week"] = df_cal.day_within_period >= 15
        df_cal["is_last_fortnight"] = df_cal.day_within_period >= 10
        return df_cal

    @staticmethod
    def construct_period_level_calendar(df_cal):
        g = df_cal.groupby("period_i")
        g = pd.DataFrame({"period_start": g.date.min(), "period_end": g.date.max()})
        return g

    def dataframes(self):
        res = []
        earnings_res = []
        for i, row in self.df.iterrows():
            df = self.fetch_history(row["symbol"], self.df_cal)
            df = df.reset_index()
            df["symbol"] = row["symbol"]
            df["is_stock"] = row["class"] == "Stock"
            res.append(df)

            if row["class"] == "Stock":
                df_earnings = self.earnings_for_ticker(row["symbol"])
                earnings_res.append(df_earnings)

        df = pd.concat(res, ignore_index=True)
        df = pd.merge(df, self.g_cal, left_on="period_i", right_index=True, how="left")

        # normalize the recent returns and recent vol by the symbol's stats
        df["log_d5_std_lag1"] = np.log(df.d5_std_lag1)
        g = df.groupby("symbol")
        g = pd.DataFrame(
            {
                "d5_r_lag1_mu": g.d5_r_lag1.mean(),
                "d5_r_lag1_std": g.d5_r_lag1.std(),
                "d10_r_lag1_mu": g.d10_r_lag1.mean(),
                "d10_r_lag1_std": g.d10_r_lag1.std(),
                "log_d5_std_lag1_mu": g.log_d5_std_lag1.mean(),
                "log_d5_std_lag1_std": g.log_d5_std_lag1.std(),
            }
        )
        df = pd.merge(df, g, left_on="symbol", right_index=True, how="outer")
        df["d5_r_lag1_z"] = (df.d5_r_lag1 - df.d5_r_lag1_mu) / df.d5_r_lag1_std
        df["d10_r_lag1_z"] = (df.d10_r_lag1 - df.d10_r_lag1_mu) / df.d10_r_lag1_std
        df["d5_std_lag1_z"] = (
            df.log_d5_std_lag1 - df.log_d5_std_lag1_mu
        ) / df.log_d5_std_lag1_std
        df_earnings = pd.concat(earnings_res, ignore_index=True)
        df.period_i = df.period_i.astype(int)
        return df, df_earnings, self.g_cal[self.g_cal.index >= 0]
