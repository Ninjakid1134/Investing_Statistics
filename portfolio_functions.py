import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import yfinance
import yfinance as yf
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from pandas import read_csv
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import stats
import warnings


def annualised_returns(stocks, n_periods, year):
    """Returns annualised returns """

    results = []

    if "" in stocks:
        stocks = [stocks]

    for stock in stocks:
        ticker = yf.download(stock, interval="1mo")["Close"].pct_change()

        ticker = ticker.loc[f"{year}":]

        obvs = ticker.shape[0]

        compound_growth = (1 + ticker).prod()

        annualise_returns = (compound_growth ** (n_periods / obvs) - 1)

        annualised_data = stock, annualise_returns

        results.append(annualised_data)

    annual = pd.DataFrame(results, columns=None)

    return annual


def cornish_var(asset, level=5, year=2020, modified=False):
    result = []

    data = []

    if "" in asset:
        asset = [asset]

    for stocks in asset:
        r = yf.download(stocks, interval="1mo")["Close"].pct_change().dropna()

        r = r.loc[f"{year}":]

        z = norm.ppf(level / 100)

        s = stats.skew(r)
        k = stats.kurtosis(r)
        z = (z +
             (z ** 2 - 1) * s / 6 +
             (z ** 3 - 3 * z) * (k - 3) / 24 -
             (2 * z ** 3 - 5 * z) * (s ** 2) / 36
             )

        var = -(r.mean() + z * r.std(ddof=0))

        datal = stocks, var.tolist()

        result.append(datal)

    for x, y in result:
        for i in y:
            data.append({"Ticker": x,
                         "Value at Risk": i,
                         })

    df = pd.DataFrame(data, index=range(1, len(data) + 1))

    total = df["Value at Risk"].mean()

    new_row = {"Ticker": "Average VAR",
               "Value at Risk": total
               }

    var = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return print(var)


def yearly_stock_returns(tickers, year):
    data = pd.DataFrame()

    for picks in tickers:
        returns = yf.download(picks, period="max", interval="1mo")["Close"].pct_change().dropna()
        data[picks] = returns.loc[f"{year}":]

    df_results = pd.DataFrame(data).dropna()

    df_results.to_csv("C:/Users/User/OneDrive/Documents/yearly_stock_returns.csv", index=False)
    results = read_csv("C:/Users/User/OneDrive/Documents/yearly_stock_returns.csv")

    return pd.DataFrame(results)


def portfolio_return(ticker, year):
    if "" in ticker:
        ticker = [ticker]

    er = annualised_returns(ticker, 12, f"{year}")

    rets = er["Annualised Return"].values

    n = len(ticker)

    weights = 1 / n

    result = round(sum(rets * weights), 2)

    return result


def portfolio_volatility(ticker, year):
    if "" in ticker:
        ticker = [ticker]

    er = yearly_stock_returns(ticker,f"{year}")

    cov_matrix = er.cov()

    n = len(ticker)

    weights = np.repeat(1 / n, n)

    result = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    return result


def fcf_average(ticker):
    result = []

    if "" in ticker:
        ticker = [ticker]

    for tickers in ticker:
        info = yf.Ticker(tickers)

        fcf = info.cashflow.loc["Free Cash Flow"]

        operating_cf = info.cashflow.loc["Operating Cash Flow"]

        fcf_pc = (fcf / operating_cf)

        fcf_pc = fcf_pc.mean()

        result.append(round(fcf_pc, 2))
    result = float(result[0])
    return result

def pe_ratio(ticker):
    result = []

    if "" in ticker:
        ticker = [ticker]

    for tickers in ticker:
        info = yf.Ticker(tickers)

        market_cap = info.info.get('marketCap')
        outstanding_shares = info.balancesheet.loc["Capital Stock"].iloc[0]
        market_per_share = market_cap / outstanding_shares

        eps = info.income_stmt.loc["Basic EPS"].iloc[0]

        if eps is None:
            eps = info.quarterly_income_stmt.loc["Basic EPS"].iloc[0]

        pe_ratios = market_per_share / eps

        result.append(round(pe_ratios, 2))
    result = float(result[0])
    return result


def pb_ratio(ticker):
    result = []

    if "" in ticker:
        ticker = [ticker]

    for tickers in ticker:
        info = yf.Ticker(tickers)

        market_cap = info.info.get('marketCap')
        total_liabs = info.balancesheet.loc["Total Liabilities Net Minority Interest"].iloc[0]
        total_asset = info.balancesheet.loc["Total Assets"].iloc[0]

        bve = total_asset - total_liabs

        pb_ratios = market_cap / bve

        result.append(round(pb_ratios, 2))
    result = float(result[0])
    return result


def cagr_eps(ticker):
    result = []

    if "" in ticker:
        ticker = [ticker]

    for tickers in ticker:
        info = yf.Ticker(tickers)

        eps_data = info.income_stmt.loc["Basic EPS"]

        if eps_data is None:
            eps_data = info.quarterly_income_stmt.loc["Basic EPS"].iloc[0]

        eps_growth = ((eps_data.iloc[0] / eps_data.iloc[-1]) ** 1 / len(eps_data))

        result.append(round(eps_growth, 2))

    result = float(result[0])
    return result


def cagr_revenue(ticker):
    result = []

    if "" in ticker:
        ticker = [ticker]

    for tickers in ticker:
        info = yf.Ticker(tickers)

        total_revenue = info.income_stmt.loc["Total Revenue"]

        total_revenue = total_revenue

        CAGR = ((total_revenue.iloc[0] /
                 total_revenue.iloc[-1]) ** 1 / len(total_revenue))

        result.append(round(CAGR, 2))

    result = float(result[0])

    return result


def roic_average(ticker):
    result = []

    if "" in ticker:
        ticker = [ticker]

    for tickers in ticker:
        info = yf.Ticker(tickers)

        invested_cap = info.balancesheet.loc["Invested Capital"]

        no_pat = (info.income_stmt.loc["EBIT"] * (1 - info.income_stmt.loc["Tax Rate For Calcs"].mean()))

        roic = invested_cap / no_pat

        roic = roic

        roic_mean = (sum(roic) / len(roic))

        roic_mean = (roic_mean / 100)

        result.append(round(roic_mean, 2))
    result = float(result[0])
    return result

def dividend_pay(ticker):
    results = []

    if "" in ticker:
        ticker = [ticker]

    for tickers in ticker:

        stock = yf.Ticker(tickers)

        if "Cash Dividends Paid" in stock.cashflow.index:

            dividend = abs(stock.cashflow.loc["Cash Dividends Paid"].iloc[0])

            oshares = stock.balance_sheet.loc["Ordinary Shares Number"].iloc[0]

            DPS = dividend / oshares

            QDPS = DPS / 4

            results.append(QDPS)

        else:
            results.append(0)

    results = sum(results)

    return print(results)

def mu(tickers, year):

    result = []

    if "" in tickers:
        tickers = [tickers]

    for ticker in tickers:
        ticker = yf.download(ticker,interval="1mo", period="max")["Close"]

        ticker = ticker.loc[f"{year}":]

        log_returns = np.log(ticker/ticker.shift(1)).dropna()

        result.append(log_returns.mean())

    mean = []

    for i in result:
        mean.append(i.values)

    rets = np.mean(mean) * 12

    return rets


def active_instrinsic_value(tickers):
    result = []

    if "" in tickers:
        tickers = [tickers]

    for ticker in tickers:
        asset_info = yf.Ticker(ticker)

        total_revenue = asset_info.quarterly_income_stmt.loc["Total Revenue"]

        eps = asset_info.income_stmt.loc["Basic EPS"].iloc[0]

        growth_rate = ((total_revenue.iloc[0] - total_revenue.iloc[1]) / total_revenue.iloc[1]) * 100

        active_forecasted_intrinsic_val = eps * (asset_info.info.get("forwardPE") +
                                                 (2 * growth_rate)) if asset_info.info.get(
            "forwardPE") is not None else None

        result.append(round(active_forecasted_intrinsic_val, 2))

    return result

def instrinsic_value(tickers):
    result = []

    if "" in tickers:
        tickers = [tickers]

    for ticker in tickers:
        asset_info = yf.Ticker(ticker)

        total_revenue = asset_info.income_stmt.loc["Total Revenue"]

        eps = asset_info.income_stmt.loc["Basic EPS"].iloc[0]

        if eps is None:
            eps = asset_info.quarterly_income_stmt.loc["Basic EPS"].iloc[0]


        growth_rate = ((total_revenue.iloc[0] - total_revenue.iloc[1]) / total_revenue.iloc[1]) * 100

        active_forecasted_intrinsic_val = eps * (asset_info.info.get("forwardPE") +
                                                 (2 * growth_rate)) if asset_info.info.get(
            "forwardPE") is not None else None

        result.append(round(active_forecasted_intrinsic_val, 2))

    result = float(result[0])

    return result

def stock_picks(tickers):
    results = []

    if "" in tickers:
        tickers = [tickers]

    for ticker in tickers:

        try:
            stocks = yf.Ticker(ticker)
            stocks_d = yf.download(ticker, interval="1mo")
            income_statement = stocks.financials
            warnings.simplefilter(action='ignore', category=FutureWarning)

            if income_statement is not None and not income_statement.empty:

                intrinsic_v = instrinsic_value(ticker)

                pe = pe_ratio(ticker)

                eps = cagr_eps(tickefr)

                pb = pb_ratio(ticker)

                fcf = fcf_average(ticker)

                rev_growth = cagr_revenue(ticker)

                roic = roic_average(ticker)

                # ROE average
                Return_on_equity = (income_statement.loc["Net Income"] /
                                    stocks.balancesheet.loc["Stockholders Equity"]).dropna()

                average_roe = Return_on_equity.mean()

                # Annualised Returns
                obvs = stocks_d.shape[0]

                compound_growth = (1 + stocks_d["Close"].pct_change()).prod()

                annualise_returns = (compound_growth ** (12 / obvs) - 1)

                # Debt to Equity
                debt = stocks.balance_sheet.loc["Total Debt"]
                equity = stocks.balancesheet.loc["Stockholders Equity"]
                debt_to_equity = debt / equity
                debt_to_equity = debt_to_equity.mean()

                results.append({
                    "Ticker": ticker,
                    "Intrinsic Value":
                        intrinsic_v,
                    "Current Price": round(float(round(stocks_d["Close"].iloc[-1])), 2),
                    f"Yearly average Return": round(annualise_returns[0], 2),
                    f"EPS growth rate": eps,
                    f"Average ROE": round(average_roe, 2),
                    f"Revenue Growth": rev_growth,
                    "Average debt to equity": round(debt_to_equity, 2),
                    "Price to Earnings ratio": pe,
                    "Price to Book ratio": pb,
                    "Return on Investment Capital": roic,
                    "Free Cash Flow": fcf,
                    f"Industry": stocks.info.get("industry")
                    if stocks.info.get("industry") is not None else None,
                    f"Business Summary": stocks.info.get("longBusinessSummary")
                    if stocks.info.get("longBusinessSummary") is not None else None,
                })
            else:
                print(f"No financial data found for {ticker}.")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    final_Df = pd.DataFrame(results)

    final_Df = final_Df.sort_values(by="Industry")

    return final_Df.to_csv("C:/Users/User/OneDrive/Documents/Stock_Valuation.csv", index=False)


def annual_rets(tickers):
    ua = UserAgent()

    header = {"User-Agent": ua.random}

    info = []

    tick = []
    try:
        for ticker in tickers:
            url = f"https://uk.finance.yahoo.com/quote/{ticker}/history/"

            response = requests.get(url, headers=header)

            website = response.content

            site = BeautifulSoup(website, "html5lib")

            body = site.find("table")

            headers = body.find_all("th")

            try:

                col = [heads.text.strip() for heads in headers]

                col[4] = "Close"

                col[5] = "Adj Close"

            # columns = pd.DataFrame(columns=col)

                rowss = body.find_all("tr")

                boat = []

                for rows in rowss:
                    row_data_2 = rows.find_all("td")
                    individual_row_data_2 = [rows.text.strip() for rows in row_data_2]
                    if len(individual_row_data_2) == len(col):
                        boat.append(individual_row_data_2)
                        if len(boat) == 13:
                            data = pd.DataFrame(boat, columns=col)

                            data = data[::-1]

                            data["Close"] = data["Close"].str.replace(",", "")

                            data["Close"] = data["Close"].astype(float)

                            close = data["Close"].pct_change().dropna()

                            obvs = close.shape[0]

                            compound_growth = (1 + close).prod()

                            annual_returns = (compound_growth ** (12 / obvs) - 1)

                            annual_data = annual_returns

                            info.append(annual_data)

                            tick.append(ticker)

                            time.sleep(2)
            except IndexError:
                pass
    except AttributeError or IndexError or None:
        pass

    returns = pd.DataFrame(tick, columns=["Ticker Symbol"])
    returns["Annual Returns"] = info

    returns.to_csv("C:/Users/User/OneDrive/Documents/Yahoo_Finance_Returns.csv", index=False)

    return returns