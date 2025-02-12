from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import os

API_KEY = 'PK9E85AK8QRJVNZC7UYZ'
SECRET_KEY = 'T6uSv4lkOPvlMBKS1Q5OGGWdbaCtuaLR62MzLmpq'

def initialize_client():
    return StockHistoricalDataClient(API_KEY, SECRET_KEY)

def check_dir(path_dir:str) -> None:
    os.makedirs(path_dir, exist_ok=True)

def save_bar_data(symbol:str, timeframe, start:str, end:str, save_dir:str) -> None:

    check_dir(save_dir) # Create directory if it doesn't exist

    # Define API request
    client = initialize_client()
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start, 
        end=end)
    
    print(f"\nRetrieving data for: {symbol} from {start} to {end}...")
    bars = client.get_stock_bars(request)
    bars = bars.df

    bars.reset_index(level='symbol', drop=True, inplace=True) # Remove multi-index

    bars.to_csv(f'./{save_dir}/{symbol}_{timeframe}_{start}_{end}.csv')
    print(f"Data saved to: {save_dir}/{symbol}_{timeframe}_{start}_{end}.csv\n")


save_bar_data('INTC', TimeFrame.Minute, '2023-08-01', '2024-01-31', 'data')