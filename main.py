import pandas as pd
import os
from dotenv import load_dotenv
import warnings
from company_data_extractor import company_data_extractor


# Register API for Financial Modeling Prep (Financial Statements and Company Fundamentals)
# https://site.financialmodelingprep.com/developer/
# Register API for Federal Reserve Economic Data (For Macroeconomics Data)
# https://fred.stlouisfed.org/docs/api/fred/


warnings.filterwarnings('ignore')

load_dotenv('.env')
API_KEY_FRED = os.environ.get('API_KEY_FRED')
API_KEY_FMP = os.environ.get('API_KEY_FMP')

start_year = 2012
end_year = 2021
num_of_years = end_year - start_year + 1

# Scrap sp500 tickers using pandas datareader
tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
ticker_table = tables[0]
tickers = ticker_table['Symbol'].tolist()

# Obtain our dataset
data_extractor = company_data_extractor(API_KEY_FRED, API_KEY_FMP)
dataset = []
company_number = 1
for ticker in tickers:
    print(f"{company_number}: Obtaining data for {ticker}")
    company_number = company_number + 1
    company_data = data_extractor.get_data(ticker, start_year, end_year, num_of_years)
    if type(company_data).__name__ == "int":
        continue
    dataset.append(company_data)
dataset = pd.concat(dataset, ignore_index=True)

# Save data to disk
dataset.to_csv("Stock_data.csv", index=False)









