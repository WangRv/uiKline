# encoding:UTF-8
import tushare as ts
import datetime as dt
from vnpy.trader.object import \
    (BarData, ContractData, Exchange, Product, Interval)
from vnpy.trader.database import mongo_manager

local_symbol_to_contract = {}
gateway_name = "ASHARE"


def down_stock_codes() -> list:
    """get all stock codes"""
    stocks = ts.get_day_all()
    stocks_contract_list = []
    for stock in stocks.itertuples():
        symbol: str = stock.code
        exchange: Exchange = Exchange.ASHARE
        name: str = stock.name
        pricetick: float = 0.01
        size: int = 100
        product: Product = Product.EQUITY
        history_data = True
        gateway_name = None
        contract = ContractData(symbol=symbol, exchange=exchange,
                                name=name, pricetick=pricetick, size=size,
                                product=product, history_data=history_data,
                                gateway_name=gateway_name)

        stocks_contract_list.append(contract)
        local_symbol_to_contract[symbol] = contract
    return stocks_contract_list


def down_stock_data(contract: ContractData) -> list:
    symbol = contract.symbol
    stock_history_data = ts.get_hist_data(symbol)
    # package to bar data object
    bar_list = []

    for stock in reversed(list(stock_history_data.itertuples())):
        date_time = dt.datetime.strptime(stock.Index, "%Y-%M-%d")
        exchange = "ASHARE"
        # price
        open_price = stock.open
        high_price = stock.high
        close_price = stock.close
        low_price = stock.low
        volume = stock.volume
        # other features
        price_change = stock.price_change
        p_change = stock.p_change
        ma5 = stock.ma5
        ma10 = stock.ma10
        ma20 = stock.ma20
        v_ma5 = stock.v_ma5
        v_ma10 = stock.v_ma10
        v_ma20 = stock.v_ma20
        bar_data = BarData(gateway_name, symbol, exchange, datetime=date_time, interval=Interval.DAILY,
                           volume=volume, open_price=open_price, high_price=high_price, low_price=low_price,
                           close_price=close_price)
        # other

        bar_list.append(bar_data)
    return bar_list


def down_load_hs300_stock_codes() -> list:
    hs300_df = ts.get_hs300s()
    stock_codes_list: list = []
    for stock in hs300_df.itertuples():
        stock_codes_list.append(stock.code)
    return stock_codes_list


def down_load_sz50_stock_codes() -> list:
    sz50_df = ts.get_sz50s()
    stock_code_list: list = []
    for stock in sz50_df.itertuples():
        stock_code_list.append(stock.code)
    return stock_code_list


def save_bar_to_mongo(bar_list: list):
    mongo_manager.save_bar_data(bar_list)


def symbol_to_contract(contracts: list) -> dict:
    stock_dict_map = {}
    for contract in contracts:
        #
        stock_dict_map[contract.symbol] = contract
    return stock_dict_map


def save_all_stock_data():
    all_stcok_contract = down_stock_codes()
    for contract in all_stcok_contract:
        stock_data_list = down_stock_data(contract)
        # save
        save_bar_to_mongo(stock_data_list)


def filter_hs300_contracts(hs300_codes: list, all_stock_contracts: dict) -> dict:
    hs300_contracts: dict = dict()
    for code in hs300_codes:
        hs300_contracts[code] = all_stock_contracts[code]
    return hs300_contracts

def filter_sz50_contracts(sz50_codes: list, all_stock_contracts: dict) -> dict:
    sz50_contracts:dict = dict()
    for code  in sz50_codes:
        sz50_contracts[code] = all_stock_contracts[code]
    return sz50_contracts

def extract_hs300_contracts() -> dict:
    all_stock_contract = down_stock_codes()
    stocks_dict = symbol_to_contract(all_stock_contract)
    hs300_codes = down_load_hs300_stock_codes()
    hs300_contracts = filter_hs300_contracts(hs300_codes, stocks_dict)
    return hs300_contracts

def extract_sz50_contracts()->dict:
    all_stock_contract = down_stock_codes()
    stocks_dict = symbol_to_contract(all_stock_contract)
    sz50_codes = down_load_sz50_stock_codes()
    sz50_contracts = filter_sz50_contracts(sz50_codes,stocks_dict)
    return sz50_contracts

if __name__ == '__main__':
    print(extract_sz50_contracts())
