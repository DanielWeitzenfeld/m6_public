from datetime import datetime
from abc import ABCMeta
import pytz

import pymc as pm
import numpy as np
import pytensor.tensor as T
import pandas as pd
from matplotlib import pyplot as plt

utc = pytz.UTC


class Model(metaclass=ABCMeta):

    def __init__(self, df, df_earnings, df_cal, weight_cutoff=datetime(2017, 1, 1)):
        self.traceplot_vars = []
        self.coords = {"asset_class": ["stock", "etf", "etf_fi"]}
        self.forecast_period = df_cal.index.max()
        df = self.prepare_data_for_model(df, weight_cutoff=weight_cutoff)
        df = self.define_earnings_variable(df, df_earnings)
        self.df = df
        self.df_cal = df_cal
        self.coords["periods"] = df_cal.period_end.values

    def define_model(self):
        pass

    def prepare_data_for_model(self, df, weight_cutoff):
        df = self.define_symbol_index(df)
        df = self.define_observation_weights(df, cutoff=weight_cutoff)
        df["d20_r_with_forecast_period"] = df.d20_r
        df.loc[df.period_i == self.forecast_period, "d20_r"] = np.NAN
        df["is_observed"] = df.d20_r.notnull()
        return df

    def define_symbol_index(self, df):
        df_symbols = (
            df[["symbol", "is_stock"]]
            .drop_duplicates()
            .sort_values(by="symbol")
            .reset_index(drop=True)
        )
        df_symbols["symbol_i"] = df_symbols.index
        # make the S&P 500 id 0, so we can easily ensure it has a positive load on the factor
        df_symbols.loc[df_symbols.symbol == "IVV", "symbol_i"] = -1
        df_symbols = df_symbols.sort_values(by="symbol_i").reset_index(drop=True)
        df_symbols["symbol_i"] = df_symbols.index
        # df_symbols['asset_class_i'] = df_symbols.is_stock.astype(int)
        df_symbols["asset_class_alt"] = None
        df_symbols.loc[df_symbols.is_stock, "asset_class_alt"] = "stock"
        df_symbols.loc[~df_symbols.is_stock, "asset_class_alt"] = "ETF"
        # some special cases
        fixed_income_etfs = [
            "LQD",
            "HYG",
            "SHY",
            "IEF",
            "TLT",
            "SEGA.L",
            "IEAA.L",
            "HIGH.L",
            "JPEA.L",
        ]
        df_symbols.loc[df_symbols.symbol.isin(fixed_income_etfs), "asset_class_alt"] = (
            "ETF_FI"
        )
        # VXX is very volatile, lump in with stocks
        df_symbols.loc[df_symbols.symbol == "VXX", "asset_class_alt"] = "stock"
        df_symbols.loc[df_symbols.asset_class_alt == "stock", "asset_class_i"] = 0
        df_symbols.loc[df_symbols.asset_class_alt == "ETF", "asset_class_i"] = 1
        df_symbols.loc[df_symbols.asset_class_alt == "ETF_FI", "asset_class_i"] = 2
        df_symbols.asset_class_i = df_symbols.asset_class_i.astype(int)
        df = pd.merge(
            df,
            df_symbols[["symbol", "symbol_i", "asset_class_i"]],
            on="symbol",
            how="left",
        )
        self.df_symbols = df_symbols
        self.coords["symbols"] = df_symbols.symbol.values
        return df

    def i_for_symbol(self, symbol):
        df = self.df_symbols
        return df.loc[df.symbol == symbol, "symbol_i"].values[0]

    @staticmethod
    def define_earnings_variable(df, df_earnings):
        df["has_earnings"] = False
        for i, row in df.iterrows():
            if not row["is_stock"]:
                continue
            mask = df_earnings.ticker == row["symbol"]
            mask &= df_earnings.report_date >= row["period_start"]
            mask &= df_earnings.report_date < row["period_end"]
            chk = mask.sum() > 0
            df.loc[i, "has_earnings"] = chk
        return df

    @staticmethod
    def define_observation_weights(df, cutoff):
        df["weight"] = 1.0
        df["days_until_present"] = (datetime.now() - df.period_end).apply(
            lambda x: x.days
        )
        cutoff_weight = df.loc[df.period_end >= cutoff, "days_until_present"].max()
        df.loc[df.period_end < cutoff, "weight"] = (
            1 - (df.days_until_present - cutoff_weight) / df.days_until_present.max()
        )
        return df

    def sample(self, **kwargs):
        with self.model:
            trace = pm.sample(return_inferencedata=True, init="adapt_diag", **kwargs)
        self.trace = trace

    def sample_ppc(self):
        with self.model:
            ppc = pm.sample_posterior_predictive(self.trace)
        self.ppc = ppc

    def create_submission(self):
        mask = self.df.period_i == self.df.period_i.max()
        res = []
        for i in np.arange(self.ppc["obs"].shape[0]):
            df_ = pd.DataFrame(
                {
                    "symbol_i": self.df.symbol_i[mask].values,
                    "d20_r": self.ppc["obs"][i][mask],
                }
            )
            df_["q"] = pd.qcut(df_.d20_r, 5, labels=["q0", "q1", "q2", "q3", "q4"])
            df_["sample"] = i
            res.append(df_)
        df_ = pd.concat(res, ignore_index=True)
        g = df_.groupby(["symbol_i", "q"]).size().unstack()
        g["total"] = g.sum(axis=1)
        for c in ["q0", "q1", "q2", "q3", "q4"]:
            g[c] = g[c] / g.total
        g = g.fillna(0)
        g = pd.merge(g, self.df_symbols, left_index=True, right_on="symbol_i")
        return g


class ModelPlotter:

    def plot_ppc_for_symbol(self, symbol_i):
        fig, ax = plt.subplots(figsize=(12, 4))
        mask = self.df.symbol_i == symbol_i
        ax.plot(self.df.period_end[mask], self.df.d20_r[mask], c="red")
        for i in np.random.choice(self.ppc["obs"].shape[0], 100, replace=False):
            ax.plot(
                self.df.period_end[mask], self.ppc["obs"][i][mask], alpha=0.03, c="grey"
            )
        symbol_i_ticker = self.df_symbols.loc[
            self.df_symbols.symbol_i == symbol_i, "symbol"
        ].values[0]
        ax.set_title(symbol_i_ticker, loc="left")
        return fig, ax

    def plot_forecast_period_hist_for_symbol(self, symbol_i, **kwargs):
        fig, ax = plt.subplots(figsize=(12, 4))
        mask = self.df.symbol_i == symbol_i
        mask &= self.df.period_i == self.df_cal.index.max()
        samples = self.ppc["obs"][:, mask]
        ax.hist(samples, **kwargs)
        symbol_i_ticker = self.df_symbols.loc[
            self.df_symbols.symbol_i == symbol_i, "symbol"
        ].values[0]
        ax.set_title(symbol_i_ticker, loc="left")
        return fig, ax


class ModelPPCSampleGenerator:
    @staticmethod
    def generate_ppc_sample(n_obs, posterior, i):
        obs_theta = posterior["obs_theta"].T[i].values
        obs_sd = posterior["obs_sd"].T[i].values
        obs_nu = posterior["obs_nu"].T[i].values.ravel()
        obs = np.random.default_rng().standard_t(obs_nu, size=n_obs)
        obs *= obs_sd
        obs += obs_theta

        return obs

    def sample_ppc(self):
        posterior = self.trace.posterior.stack(sample=["chain", "draw"])
        ppc = {"obs": []}
        n_obs = self.df.shape[0]
        n_samples = posterior["obs_theta"].T.shape[0]
        for i in np.arange(n_samples):
            ret = self.generate_ppc_sample(n_obs, posterior, i)
            ppc["obs"].append(ret)
        ppc["obs"] = np.array(ppc["obs"])
        self.ppc = ppc


class ModelR(ModelPPCSampleGenerator, ModelPlotter, Model):
    """
    AR, with an earnings component - partially pooled
    Observation weighting.
    Coefficient on recent returns and recent vol, partially pooled
    ability to input manual forecasts, with manually specified standard devs
    """

    def __init__(
        self,
        df,
        df_earnings,
        df_cal,
        n_factors,
        forecasts=None,
        weight_cutoff=datetime(2017, 1, 1),
    ):
        super().__init__(
            df=df, df_earnings=df_earnings, df_cal=df_cal, weight_cutoff=weight_cutoff
        )
        self.n_factors = n_factors
        self.coords["factors"] = np.arange(self.n_factors)
        self.forecasts = forecasts
        self.set_forecasts(forecasts)

    def set_forecasts(self, forecasts):
        df = self.df
        df_symbols = self.df_symbols
        df["is_manual_forecast"] = False
        df["d20_r_with_forecast"] = df.d20_r
        df["forecast_sd"] = 1
        forecast_period = self.df_cal.index.max()
        assert forecast_period == self.forecast_period
        mask = df.period_i == self.forecast_period
        if forecasts is not None:
            for symbol, forecast in forecasts.items():
                forecast_mean = forecast["mean"]
                forecast_sd = forecast["sd"]
                sym_i = df_symbols.loc[df_symbols.symbol == symbol, "symbol_i"].values[
                    0
                ]
                mask_ = mask & (df.symbol_i == sym_i)
                df.loc[mask_, "d20_r_with_forecast"] = forecast_mean
                df.loc[mask_, "forecast_sd"] = forecast_sd
                df.loc[mask_, "is_manual_forecast"] = True
        df["is_observed"] = df.d20_r_with_forecast.notnull()
        self.df = df

    def define_model(self):
        df = self.df
        df_symbols = self.df_symbols
        D = len(self.coords["symbols"])
        K = len(self.coords["factors"])
        N = len(self.coords["periods"])
        weights = df.weight.values
        is_observed = df.is_observed.values
        is_manual_forecast = df.is_manual_forecast.astype(int).values
        forecast_sd = df.forecast_sd.values

        def logp_ar(
            value: T.TensorVariable,
            mu: T.TensorVariable,
            rho: T.TensorVariable,
            sigma: T.TensorVariable,
        ) -> T.TensorVariable:
            innovations = value[1:] - rho * value[:-1]
            return T.sum(pm.logp(pm.Normal.dist(mu, sigma=sigma), innovations))

        with pm.Model(coords=self.coords) as model:
            w = pm.StudentT(
                "w",
                mu=T.zeros([D, K]),
                sigma=T.ones([D, K]),
                dims=("symbols", "factors"),
                nu=10,
            )

            k = pm.Normal("k", mu=0, sigma=0.2, initval=-0.05)

            z = pm.CustomDist("z", 0, k, 1, logp=logp_ar, dims=("periods", "factors"))

            mu = pm.Deterministic("mu", T.dot(w, z.T).T, dims=("periods", "symbols"))

            intercept_sd = pm.HalfNormal("intercept_sd", sigma=0.005)
            intercept_offset = pm.StudentT(
                "intercept_offset", mu=0, sigma=1, nu=10, dims="symbols"
            )

            beta_mu = pm.Normal("beta_mus", mu=-2, sigma=5, dims="asset_class")
            beta_mu_sd = pm.HalfNormal("beta_mu_sd", 0.3, dims="asset_class")
            beta_mu_nu = pm.Gamma("beta_mu_nu", mu=10, sigma=5)
            beta_mu_offset = pm.StudentT(
                "beta_mu_offset", mu=0, sigma=1, nu=beta_mu_nu, dims="symbols"
            )

            beta = pm.Deterministic(
                "beta",
                beta_mu[df_symbols.asset_class_i.values]
                + beta_mu_offset[df_symbols.symbol_i.values]
                * beta_mu_sd[df_symbols.asset_class_i.values],
                dims="symbols",
            )
            beta_earnings_mu = pm.Normal("beta_earnings_mu", mu=0, sigma=1)
            beta_earnings_offset = pm.Normal(
                "beta_earnings_offset", mu=0, sigma=1, dims="symbols"
            )
            beta_earnings_sd = pm.HalfNormal("beta_earnings_sd", 0.2)

            beta_earnings = pm.Deterministic(
                "beta_earnings",
                beta_earnings_mu
                + beta_earnings_sd * beta_earnings_offset[df_symbols.symbol_i.values],
                dims="symbols",
            )

            beta_rr_mu = pm.Normal("beta_rr_mu", mu=0.0, sigma=0.2, dims="asset_class")
            beta_rr_sd = pm.HalfNormal("beta_rr_sd", 0.3, dims="asset_class")
            beta_rr_offset = pm.StudentT(
                "beta_rr_offset", mu=0, sigma=1, nu=10, dims="symbols"
            )

            beta_recent_returns = pm.Deterministic(
                "beta_recent_returns",
                beta_rr_mu[df_symbols.asset_class_i.values]
                + beta_rr_offset[df_symbols.symbol_i.values]
                * beta_rr_sd[df_symbols.asset_class_i.values],
                dims="symbols",
            )

            obs_theta = pm.Deterministic(
                "obs_theta",
                mu[tuple(df.period_i.values), tuple(df.symbol_i.values)]
                + beta_recent_returns[df.symbol_i.values]
                * df.d5_r_lag1_z.values  # note redef of recent returns
                + intercept_sd * intercept_offset[df.symbol_i.values],
            )

            beta_recvol_mu = pm.Normal(
                "beta_recvol_mu", mu=0.05, sigma=0.2, dims="asset_class"
            )
            beta_recvol_sd = pm.HalfNormal("beta_recvol_sd", 0.3, dims="asset_class")
            beta_recvol_offset = pm.StudentT(
                "beta_recvol_offset", mu=0, sigma=1, nu=10, dims="symbols"
            )

            beta_recent_vol = pm.Deterministic(
                "beta_recent_vol",
                beta_recvol_mu[df_symbols.asset_class_i.values]
                + beta_recvol_offset[df_symbols.symbol_i.values]
                * beta_recvol_sd[df_symbols.asset_class_i.values],
                dims="symbols",
            )

            obs_sd = pm.Deterministic(
                "obs_sd",
                T.exp(
                    beta[df.symbol_i.values]
                    + beta_recent_vol[df.symbol_i.values] * df.d5_std_lag1_z.values
                    + beta_earnings[df.symbol_i.values] * df.has_earnings.values
                ),
            )

            obs_sd_w_forecast = pm.Deterministic(
                "obs_sd_w_forecast",
                T.switch(
                    T.eq(is_manual_forecast, 1),
                    forecast_sd,
                    T.exp(
                        beta[df.symbol_i.values]
                        + beta_recent_vol[df.symbol_i.values] * df.d5_std_lag1_z.values
                        + beta_earnings[df.symbol_i.values] * df.has_earnings.values
                    ),
                ),
            )

            obs_nu_sd = pm.HalfNormal("obs_nu_sd", 4, initval=2)
            obs_nu_symbols = pm.Gamma(
                "obs_nu_symbols", mu=7, sigma=obs_nu_sd, dims="symbols"
            )
            obs_nu = pm.Deterministic("obs_nu", obs_nu_symbols[df.symbol_i.values])

            obs = pm.Potential(
                "obs",
                weights[is_observed]
                * pm.logp(
                    pm.StudentT.dist(
                        mu=obs_theta[is_observed],
                        sigma=obs_sd_w_forecast[is_observed],
                        nu=obs_nu[is_observed],
                    ),
                    df.loc[is_observed, "d20_r_with_forecast"].values,
                ),
            )

        self.model = model
        self.traceplot_vars = [
            "k",
            "beta_mus",
            "beta_mu_sd",
            "beta_rr_mu",
            "beta_rr_sd",
            "beta_recvol_mu",
            "beta_recvol_sd",
            "beta_recent_returns",
            "beta_recent_vol",
            "beta_earnings_mu",
            "beta_earnings_sd",
            "intercept_sd",
            "obs_nu_sd",
        ]

    @staticmethod
    def generate_ppc_sample(n_obs, posterior, i):
        obs_theta = posterior["obs_theta"].T[i].values
        obs_sd = posterior["obs_sd"].T[i].values
        obs_nu = posterior["obs_nu"].T[i].values
        obs = np.random.default_rng().standard_t(obs_nu, size=n_obs)
        obs *= obs_sd
        obs += obs_theta

        return obs


name_to_class = {
    "R": ModelR,
}
