import numpy as np
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt
import tushare as ts


# Kalman滤波函数
def kalmanfilter(a, u, v):  # a为Series格式
    print("正在处理Kalman滤波...")
    size = len(a)
    x = np.zeros(size)
    z = np.zeros(size)

    p = np.zeros(size)
    q = np.zeros(size)

    p[0] = u * u
    q[0] = u * u
    x[0] = a[0]
    z[0] = a[0]
    for i in range(1, size):
        x[i] = z[i - 1]
        q[i] = p[i - 1] + u * u
        lam = q[i] / (q[i] + v * v)
        # z[i] = (1-lam) * x[i-1] + lam * a[i] # 此为错误代码
        z[i] = (1 - lam) * x[i] + lam * a[i]  # 此为正确代码
        p[i] = q[i] * v * v / (q[i] + v * v)

    z_ = a.copy()
    z_[:] = z
    return z_


# 滤波处理数据函数
def data_filter(data, filter=False):
    if isinstance(data, pd.Series):
        data_close = data
    elif isinstance(data, pd.DataFrame):
        try:
            data_close = data.loc[:, "close"]
        except:
            print("滤波处理中，但DataFrame中没有close列标签")

    if filter:
        print("正在Kalman滤波处理...")
        u = 1
        v = 3
        data_close_filtered = kalmanfilter(data_close, u, v)
    else:
        data_close_filtered = data_close

    if isinstance(data, pd.DataFrame):
        data_filtered = data
        data_filtered.loc[:, "close"] = data_close_filtered.values
    else:
        data_filtered = data_close_filtered

    return data_filtered


# 生成交易信号函数
def trade_sigal(data_filtered, factor, set_para, frequency, window=30):
    # 数据格式化
    if isinstance(data_filtered, pd.Series):
        data_close_filtered = data_filtered
    elif isinstance(data_filtered, pd.DataFrame):
        try:
            data_close_filtered = data_filtered.loc[:, "close"]
        except:
            print("回测中, DataFrame中没有close列标签。")
    # 计算因子值
    if factor == "ad_ma":
        factor_value_list = ad_ma_factor(data_filtered)
    elif factor == "rsi":
        factor_value_list = rsi_factor(data_close_filtered)

    print("正在生成交易信号...")
    trade_signal = pd.Series(np.zeros(len(data_filtered)), index=data_filtered.index)

    # 生成仓位信号
    for n in range(window, len(data_filtered), frequency):
        factor_value = factor_value_list[n]

        # 若上一时刻为空仓/未持仓，且factor_value < -set_para
        # 则接下来开多仓
        if trade_signal[n - 1] <= 0 and factor_value < -set_para:
            trade_signal[n: n + frequency] = 1

        # 若上一时刻为多仓/未持仓，且factor_value > set_para
        # 则接下来开空仓
        elif trade_signal[n - 1] >= 0 and factor_value > set_para:
            trade_signal[n: n + frequency] = -1

        # 若上一时刻为多仓，且 -set_para < factor_value < set_para
        # 则接下来仍持有多仓
        elif trade_signal[n - 1] == 1 and -set_para < factor_value < set_para:
            trade_signal[n: n + frequency] = 1

        # 若上一时刻为空仓，且 -set_para < factor_value < set_para
        # 则接下来仍持有空仓
        elif trade_signal[n - 1] == -1 and -set_para < factor_value < set_para:
            trade_signal[n: n + frequency] = -1
    return trade_signal


# 回测函数，根据交易信号生成账户净值
def calculator(data, trade_signal, initial_value, expense=0, window=30):
    print("正在回测计算账户净值...")
    # 初始化
    if isinstance(data, pd.Series):
        data_close = data
    elif isinstance(data, pd.DataFrame):
        try:
            data_close = data.loc[:, "close"]
        except:
            print("回测中, DataFrame中没有close列标签。")
    trade_count = 0  # 记录总开关仓次数
    net_value = pd.Series(index=data_close.index, dtype="float64")  # 记录当日账户价值走势
    net_value[:window + 1] = initial_value
    # print("trade_signal: ", trade_signal[window])

    # 循环回测，根据仓位信号序列生成账户价值走势
    for n in range(window, len(trade_signal) - 1):
        # 设定每次开仓的价值
        temp_pos = net_value[n]

        # 如果信号为开多仓
        if trade_signal[n] == 1:
            net_value[n + 1] = net_value[n] + temp_pos * (data_close[n + 1] / data_close[n] - 1)
            temp_pos = temp_pos * data_close[n + 1] / data_close[n]

        # 如果信号为开空仓
        elif trade_signal[n] == -1:
            net_value[n + 1] = net_value[n] + temp_pos * (1 - data_close[n + 1] / data_close[n])
            temp_pos = temp_pos * (2 - data_close[n + 1] / data_close[n])

        # 如果信号为不持仓
        else:
            net_value[n + 1] = net_value[n]
            temp_pos = 0  # 重新初始化为10万元

        # 如果本次持仓与上一时刻持仓不同，则扣除手续费（例如，从多仓到不持仓，扣除1笔；从多仓到空仓，扣除2笔）
        if trade_signal[n] != trade_signal[n - 1]:
            trade_count = trade_count + np.abs(trade_signal[n] - trade_signal[n - 1])  # 更新调仓次数
            net_value[n + 1] = net_value[n + 1] - expense * temp_pos * np.abs(
                trade_signal[n] - trade_signal[n - 1])  # 扣除手续费

    return net_value, trade_count  # 输出总账户价值走势、平均每日开关仓次数


# 绘图函数
def draw_fig(
        data, net_value, code, factor,
        initial_value=10, expense=True
):
    # 初始化
    pro = ts.pro_api()
    df = pro.stock_basic(**{
        "ts_code": code,
    }, fields=[
        "name"
    ])
    name = df.iloc[0, 0]

    data.index = pd.to_datetime(data.index, format= "%Y%m%d")
    net_value.index = pd.to_datetime(net_value.index, format="%Y%m%d")
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    if isinstance(data, pd.Series):
        data_close = data
    elif isinstance(data, pd.DataFrame):
        try:
            data_close = data.loc[:, "close"]
        except:
            print("绘图中, DataFrame中没有close列标签。")
    base_value = data_close / data_close.values[0] * initial_value

    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.grid()
    ax.plot(base_value.index, base_value.values, label=str(name) + "价格走势")
    ax.plot(net_value.index, net_value.values, label="账户净值走势")
    ax.set_ylabel('净值')
    ax.set_xlabel('时间')
    if expense:
        plt.title("基于" + factor.upper() + '因子择时交易')
    elif not expense:
        plt.title("基于" + factor.upper() + '因子择时交易')
    ax.legend()
    plt.show()
    plt.close()


# P&L评判指标输出函数
def print_pl(net_value, orders, expense, filter = "None"):
    # 初始化
    global trade_days
    if expense:
        print('\n双边' + str(expense * 100) + '%手续费、' + str(filter) + '滤波: ')
    elif not expense:
        print('\n0手续费、' + str(filter) + '滤波: ')

    # 计算收益及标准差
    return_index = net_value[1:].index
    temp_value = net_value[:-1]
    next_value = net_value[1:]
    temp_value.index = list(range(0, len(net_value.index) - 1))
    next_value.index = list(range(0, len(net_value.index) - 1))
    return_none = (next_value - temp_value) / temp_value
    return_none.index = return_index
    total_return = (net_value[-1] - net_value[0]) / net_value[0]
    annual_return = total_return * 255 / len(trade_days.index)
    std_None = np.std(return_none) * np.sqrt(len(return_none))  # return_none.std()
    # if expense:
    #     return_none.to_csv('双边' + str(expense * 100) + '%手续费、' + str(filter) + '滤波' + ".csv", encoding="gbk")
    # elif not expense:
    #     return_none.to_csv('0手续费、' + str(filter) + '滤波' + ".csv", encoding="gbk")

    # 计算最大回撤
    arry = pd.Series(net_value)
    cumlist = arry.cummax()
    max_drawdown = np.max((cumlist - arry) / cumlist)
    # 输出收益结果
    print("年化收益率：{:.2%}".format(annual_return))
    print('总收益率：{:.2%}'.format(total_return))
    print('标准差：{:.2f}'.format(std_None))
    print('夏普比率：{:.2f}'.format((total_return - 0.03 * 255 / len(trade_days)) / std_None))
    print('最大回撤：{:.2%}'.format(max_drawdown))
    print('总开仓次数：{:.0f}'.format(orders / 2))
    print('日均开仓次数：{:.2f}'.format(orders / 2 / len(trade_days)))


# 构造技术指标因子，已添加支持的技术因子：
# 重叠指标因子：EMA、BOLL
# 动量因子：RSI、KDJ、MACD、WILLR
# 量价指标：AD、AD_MA、ADOSC、ADOSC_MA、OBV

# 构造EMA因子
def ema_factor(data_close, fast_period=10, slow_period=20):
    print("正在构造并计算因子ema_factor...")
    ema_fast = ta.EMA(data_close, fast_period)
    ema_slow = ta.EMA(data_close, slow_period)
    ema_factor = ema_fast - ema_slow
    ema_factor = (ema_factor - ema_factor.mean()) / ema_factor.std()
    return ema_factor


# 构造boll因子
def boll_factor(data_close, timeperiod=20, nbdevup=2, nbdevdn=2):
    print("正在构造并计算因子boll_factor...")
    boll_up, boll_mid, boll_dn = ta.BBANDS(data_close, timeperiod, nbdevup, nbdevdn)
    boll_std = (boll_up - boll_dn) / (nbdevup + nbdevdn)
    boll_factor = (data_close - boll_mid) / boll_std
    return boll_factor


# 构造RSI因子
def rsi_factor(data_close, period=20):
    print("正在构造并计算因子rsi_factor...")
    rsi = ta.RSI(data_close, period)
    rsi_factor = (rsi - rsi.mean()) / rsi.std()
    return rsi_factor


# 构造KDJ因子
def kdj_factor(data_close, ma=20, period1=3, period2=2):
    print("正在构造并计算因子kdj_factor...")
    high = data_close.rolling(ma).max()
    low = data_close.rolling(ma).min()
    kdj_k, kdj_d = ta.STOCH(high, low, data_close)
    kdj_j = period1 * kdj_k - period2 * kdj_d
    kdj_k = (kdj_k - kdj_k.mean()) / kdj_k.std()
    kdj_d = (kdj_d - kdj_d.mean()) / kdj_d.std()
    kdj_j = (kdj_j - kdj_j.mean()) / kdj_j.std()
    return kdj_k, kdj_d, kdj_j


# 构造MACD因子
def macd_factor(data_close, fast_period=18, slow_period=20, ma=9):
    print("正在构造并计算因子macd, diff, dea...")
    macd, diff, dea = ta.MACD(
        data_close, fast_period, slow_period, ma
    )
    macd = (macd - macd.mean()) / macd.std()
    diff = (diff - diff.mean()) / diff.std()
    dea = (dea - dea.mean()) / diff.std()
    return macd, diff, dea


# 构造WILLR指标因子
def willr_factor(data_close, timeperiod=14):
    print("正在构造并计算因子willr_factor...")
    high = data_close.rolling(timeperiod).max()
    low = data_close.rolling(timeperiod).min()
    willr = ta.WILLR(high, low, data_close, timeperiod=timeperiod)
    willr = (dwillr - willr.mean()) / willr.std()
    return willr


# 构造AD指标因子
def ad_factor(data):
    print("正在构造并计算因子ad_factor...")
    high = data.loc[:, "high"]
    low = data.loc[:, "low"]
    close = data.loc[:, "close"]
    volume = data.loc[:, "amount"]
    ad_factor = ta.AD(high, low, close, volume)
    ad_factor = (ad_factor - ad_factor.mean()) / ad_factor.std()
    return ad_factor


# 构造AD_MA指标因子
def ad_ma_factor(data_in, ma=10):
    print("正在构造并计算因子ad_ma_factor...")
    high = data_in.loc[:, "close"].rolling(ma).max()
    low = data_in.loc[:, "close"].rolling(ma).min()
    close = data_in.loc[:, "close"]
    volume_ma = data_in.loc[:, "vol"]
    ad_ma = ta.AD(high, low, close, volume_ma)
    ad_ma = (ad_ma - ad_ma.mean()) / ad_ma.std()
    return ad_ma


# 构造ADOSC因子
def adosc_factor(data, fast_period=3, slow_period=10):
    print("正在构造并计算因子adosc_factor...")
    high = data.loc[:, "high"]
    low = data.loc[:, "low"]
    close = data.loc[:, "close"]
    volume = data.loc[:, "amount"]
    adsoc = ta.ADOSC(high, low, close, volume, fast_period, slow_period)
    adsoc = (adsoc - adsoc.mean()) / adsoc.std()
    return adsoc


# 构造ADOSC_MA因子
def adosc_ma_factor(data, fast_period=3, slow_period=10, rolling_period=10):
    print("正在构造并计算因子adosc_ma_factor...")
    high = data.loc[:, "close"].rolling(rolling_period).max()
    low = data.loc[:, "close"].rolling(rolling_period).min()
    close = data.loc[:, "close"]
    volume = data.loc[:, "vol"]
    adosc_ma = ta.ADOSC(high, low, close, volume, fast_period, slow_period)
    adosc_ma = (adosc_ma - adosc_ma.mean()) / adosc_ma.std()
    return adosc_ma


# 构造OBV因子
def obv_factor(data):
    print("正在构造并计算因子obv_factor...")
    obv = ta.OBV(data.loc[:, "close"], data.loc[:, "vol"])
    obv = (obv - obv.mean()) / obv.std()
    return obv


def get_price(code, start_date, end_date):
    # 获取交易日历
    global trade_days
    pro = ts.pro_api()
    trade_days = pro.trade_cal(**{
        "exchange": "SSE",
        "cal_date": "",
        "start_date": start_date,
        "end_date": end_date,
        "is_open": 1,
        "limit": "",
        "offset": ""
    }, fields=[
        "exchange",
        "cal_date",
        "is_open",
        "pretrade_date"
    ])
    trade_days.set_index("cal_date", inplace=True)
    # 提取价格数据
    # 拉取数据
    data = pro.daily(**{
        "ts_code": code,
        "trade_date": "",
        "start_date": start_date,
        "end_date": end_date,
        "offset": "",
        "limit": ""
    }, fields=[
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount"
    ])
    data.sort_values("trade_date", inplace=True)
    data.set_index("trade_date", inplace=True)
    data_close = data.loc[:, "close"]

    return data, data_close


# 回测程序
def backtest(
        code, start_date, end_date, factor, set_para, frequency, expense,
        filter = False, initial_value = 1
):
    # 初始化
    # 设置Token  @数据来源于开源社区，请勿用于商业用途，谢谢！
    ts.set_token("48ada6a904815ce02f7d1d4665416a8b012e0b8294f0808bdc4a840d")
    # 拉取数据并处理
    data, data_close = get_price(code, start_date, end_date)
    data_filtered = data_filter(data, filter)
    trade_signal = trade_sigal(data_filtered, factor, set_para, frequency)

    # 0%手续费， Kalman滤波
    net_value, orders = calculator(data, trade_signal, initial_value, expense)

    # 绘制P&L
    draw_fig(data, net_value, code, factor, net_value, expense)
    # 输出收益数据
    print_pl(net_value, orders, expense, filter="Kalman")


if __name__ == "__main__":
    print("程序开始：")
    backtest(
        code= "000001.SZ", start_date= "20220101", end_date= "20221208", factor= "ad_ma",
        set_para=1, frequency=10, expense=0.0002, filter=False
    )