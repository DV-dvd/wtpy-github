from wtpy import BaseHftStrategy
from wtpy import HftContext
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from datetime import datetime
import numpy as np
import pandas as pd
def makeTime(date:int, time:int, secs:int):
    '''
    将系统时间转成datetime\n
    @date   日期，格式如20200723\n
    @time   时间，精确到分，格式如0935\n
    @secs   秒数，精确到毫秒，格式如37500
    '''
    return datetime(year=int(date/10000), month=int(date%10000/100), day=date%100,
                    hour=int(time/100), minute=time%100, second=int(secs/1000), microsecond=secs%1000*1000)

class HftStraDemo(BaseHftStrategy):
    "demo中的策略"
    def __init__(self, name:str, code:str, expsecs:int, offset:int, freq:int=60, day_close = None):
        BaseHftStrategy.__init__(self, name)

        '''交易参数'''
        self.__code__ = code            #交易合约
        self.__expsecs__ = expsecs      #订单超时秒数
        self.__offset__ = offset        #指令价格偏移
        self.__freq__ = freq            #交易频率控制，指定时间内限制信号数，单位秒
        self.n = 0
        self.last_price = 0
        self.day_close = day_close               #设置是否日内平仓，若为True，则日内平仓；


        '''内部数据'''
        self.__last_tick__ = None       #上一笔行情
        self.__orders__ = dict()        #策略相关的订单
        self.__last_entry_time__ = None #上次入场时间
        self.__cancel_cnt__ = 0         #正在撤销的订单数
        self.__channel_ready__ = False  #通道是否就绪
        self.grid_change_last = [0, 0]  #记录上一次交易时网格范围的变化情况
        self.grid_change_new = []       #记录最新交易时网格范围的变化情况
        self.boll_data = pd.DataFrame(None, columns=['time', 'price', 'high', 'low', 'pre_close'])  # 用于求均线、ATR等指标
        self.last_grid = 0              #储存前一个网格所处区间，用来和最新网格所处区间作比较
        self.last_deal = {}             #记录上一笔成交的信息
        self.TR_sum = 0                 #用于求ATR指标
        self.accumulated_tick = 0       #计算累计切片数，用于估算时间，然后用于出场模块

    def on_init(self, context:HftContext):
        '''
        策略初始化，启动的时候调用\n
        用于加载自定义数据\n
        @context    策略运行上下文
        '''

        #先订阅实时数据
        context.stra_sub_ticks(self.__code__)

        self.__ctx__ = context

    def check_orders(self):
        #如果未完成订单不为空
        if len(self.__orders__.keys()) > 0 and self.__last_entry_time__ is not None:
            #当前时间，一定要从api获取，不然回测会有问题
            now = makeTime(self.__ctx__.stra_get_date(), self.__ctx__.stra_get_time(), self.__ctx__.stra_get_secs())
            span = now - self.__last_entry_time__
            if span.total_seconds() > self.__expsecs__: #如果订单超时，则需要撤单，span.total_seconds用于计算超时的总秒数
                for localid in self.__orders__:
                    self.__ctx__.stra_cancel(localid)
                    self.__cancel_cnt__ += 1
                    self.__ctx__.stra_log_text("cancelcount -> %d" % (self.__cancel_cnt__))


    def compute_boll(self,i,ma:int):
        """
        计算布林线
        i表示最新价格切片的索引
        ma=10，为10分钟均线
        windows为窗口期，单位为分钟；
        968个切片为10分钟；1965个切片为20分钟；2800个切片为30分钟；1377个切片为14分钟；121个切片是1分钟
        """
        if ma == 10:
            windows = 968
        if ma == 20:
            windows = 1965
        if ma == 30:
            windows = 2800
        if ma == 14:
            windows == 1377

        #计算窗口期的均价和标准差
        mean_price = self.boll_data['price'][i-windows-1:i].mean()
        std_data = self.boll_data['price'][i-windows-1:i].std()

        return mean_price, std_data

    def ATR(self, i, N):
        """
        计算ATR指标,N表示周期，以分钟为单位,每分钟对应的切片数看compute_boll函数
        i表示最新价格切片的索引
        """
        #一分钟的切片数大约有121个
        windows = 121
        for j in range(1,N+1):
            #每分钟滑动1次，共滑动N次
            max_price = self.boll_data['price'][i - j * windows - 1 : i - (j-1)*windows].max()
            min_price = self.boll_data['price'][i - j * windows - 1 : i - (j-1)*windows].min()
            #上一个时期的收盘价
            last_close_price = self.boll_data['price'][i - j * windows - 2]
            TR = max(max_price - min_price,abs(last_close_price - max_price),abs(last_close_price - min_price))
            self.TR_sum = self.TR_sum + TR
        ATR = self.TR_sum / N
        #每次计算完一个窗口的ATR后，将变量更新为零，以用于下一次计算
        self.TR_sum = 0
        return ATR

    def signal_with_moves(self,context:HftContext,grid):
        if self.last_grid < grid:
            self.grid_change_new = [self.last_grid,grid]
            #初始阶段，不构成信号
            if self.last_grid == 0:
                self.last_grid = grid
            else:
                if self.grid_change_new != self.grid_change_last:
                    self.last_grid = grid
                    self.grid_change_last = self.grid_change_new
                    signal = -1
                    moves = 1
                    return signal,moves
                else:
                    self.last_grid = grid
                    self.grid_change_last = self.grid_change_new
        elif self.last_grid > grid:
            self.grid_change_new = [grid,self.last_grid]
            #初始阶段，不构成信号
            if self.last_grid == 0:
                self.last_grid = grid
            else:
                if self.grid_change_new != self.grid_change_last:
                    self.last_grid = grid
                    self.grid_change_last = self.grid_change_new
                    signal = 1
                    moves = 1
                    return signal, moves
                else:
                    self.last_grid = grid
                    self.grid_change_last = self.grid_change_new
        #默认信号和开仓手数均为0，不生成信号
        return 0, 0

    def grid_with_gap(self,centre_price,gap):
        """
        通过中枢价格和网格间距生成网格
        """
        band = np.array([centre_price - 2*gap, centre_price - 1.5*gap, centre_price - 1*gap, centre_price - 0.5*gap,
                         centre_price,
                         centre_price + 0.5*gap, centre_price + 1*gap, centre_price + 1.5*gap, centre_price + 2*gap])
        return band


    def stop_loss(self,context:HftContext,curPos,newTick,ATR):
        """
        止损模块，以ATR作为止损出场指标
        """
        now = makeTime(self.__ctx__.stra_get_date(), self.__ctx__.stra_get_time(),
                       self.__ctx__.stra_get_secs())

        #判断多头已经进入损失状态
        if curPos > 0 and newTick['price'] < self.last_deal['price']:
            #此时认为价格波动较大且价格处于下跌方向
            #if ATR > 0.01 and self.__last_tick__['price'] > newTick['price']:
            if ATR > 0.01 and self.probability()>0.03:
                context.stra_log_text("强行平多止损")
                ids = context.stra_sell(self.__code__, 0, abs(curPos), "sell")
                # 将订单号加入到管理中
                for localid in ids:
                    self.__orders__[localid] = localid
                # 更新入场时间
                self.__last_entry_time__ = now
                return True

        # 判断空头已经进入损失状态
        elif curPos < 0 and newTick['price'] > self.last_deal['price']:
            # 此时认为价格波动较大且价格处于上涨方向
            if ATR > 0.01 and self.probability()>0.03:
                context.stra_log_text("强行平空止损")
                ids = context.stra_buy(self.__code__, 0, abs(curPos), "buy")
                # 将订单号加入到管理中
                for localid in ids:
                    self.__orders__[localid] = localid
                # 更新入场时间
                self.__last_entry_time__ = now
                return True

    def take_profit(self, context:HftContext, curPos, newTick, ATR):
        """
        止盈模块，以ATR作为止损出场指标
        """
        now = makeTime(self.__ctx__.stra_get_date(), self.__ctx__.stra_get_time(),
                       self.__ctx__.stra_get_secs())
        time = newTick['time']
        #多头时，现价大于成本价，则进入止盈逻辑
        if curPos > 0 and newTick['price'] > self.last_deal['price']:
            #if ATR > 0.02 and newTick['price'] < self.__last_tick__['price']: # 此时认为价格波动较大且价格朝不利方向变动
            if ATR > 0.02 and self.probability()>0.03:
                context.stra_log_text("强行平多止盈")
                ids = context.stra_sell(self.__code__, 0, abs(curPos), "sell")
                # 将订单号加入到管理中
                for localid in ids:
                    self.__orders__[localid] = localid
                # 更新入场时间
                self.__last_entry_time__ = now
                return True
        #空头时，现价小于成本价，则进入止盈逻辑
        elif curPos < 0 and newTick['price'] < self.last_deal['price']:
            #if ATR > 0.02 and newTick['price'] > self.__last_tick__['price']:  # 此时认为价格波动较大且价格朝不利方向变动
            if ATR > 0.02 and newTick['price'] > self.probability()>0.03:
                context.stra_log_text("强行平空止盈")
                ids = context.stra_buy(self.__code__, 0, abs(curPos), "buy")
                # 将订单号加入到管理中
                for localid in ids:
                    self.__orders__[localid] = localid
                # 更新入场时间
                self.__last_entry_time__ = now
                return True

    def stop_trade(self, context:HftContext, price, curPos, mean_price_MA20, ATR, time_judge):
        """
        出场模块，逻辑与入场相对应
        """
        if abs(price - mean_price_MA20) > 1.15 * ATR:
            self.accumulated_tick += 1
            # 持续时间满6分钟,进入出场逻辑,将头寸全部平掉出场
        if self.accumulated_tick >= 550:
            #将参数重置
            self.last_grid = 0
            self.grid_change_last = [0, 0]
            self.last_deal = {}
            self.__last_tick__ = None
            self.boll_data = pd.DataFrame(None, columns=['time', 'price', 'high', 'low', 'pre_close'])
            if curPos > 0:  # 持有多头头寸，此时要进行空头平仓操作
                # 当前时间，一定要从api获取，不然回测会有问题
                now = makeTime(self.__ctx__.stra_get_date(), self.__ctx__.stra_get_time(),
                               self.__ctx__.stra_get_secs())
                context.stra_log_text("平仓出场")
                ids = context.stra_sell(self.__code__, 0, abs(curPos), "sell")  # 设置为0表示以市价卖出
                # 将订单号加入到管理中
                for localid in ids:
                    self.__orders__[localid] = localid
                # 更新入场时间
                self.__last_entry_time__ = now

            elif curPos < 0:  # 持有空头头寸，此时要进行多头平仓操作
                # 当前时间，一定要从api获取，不然回测会有问题
                now = makeTime(self.__ctx__.stra_get_date(), self.__ctx__.stra_get_time(),
                               self.__ctx__.stra_get_secs())
                context.stra_log_text("平仓出场")
                ids = context.stra_buy(self.__code__, 0, abs(curPos), "buy")  # 设置为0表示以市价卖出
                # 将订单号加入到管理中
                for localid in ids:
                    self.__orders__[localid] = localid
                # 更新入场时间
                self.__last_entry_time__ = now
            # 重置参数
            if time_judge >= 1515:
                self.accumulated_tick = 0
            return True
        return False



    def daily_close(self,context:HftContext, curPos, grid, time_judge):
        """
        每日平仓模块，包含正常交易
        """
        #交易部分
        if time_judge not in self.close_timeare:
            signal, moves = self.signal_with_moves(context, grid)
            return signal,moves
        #每日平仓部分
        else:
            #将参数重置
            self.last_grid = 0
            self.grid_change_last = [0, 0]
            self.last_deal = {}
            self.__last_tick__ = None
            self.boll_data = pd.DataFrame(None, columns=['time', 'price', 'high', 'low', 'pre_close'])
            self.accumulated_tick = 0
            if curPos > 0:  # 持有多头头寸，此时要进行空头平仓操作
                # 当前时间，一定要从api获取，不然回测会有问题
                now = makeTime(self.__ctx__.stra_get_date(), self.__ctx__.stra_get_time(),
                               self.__ctx__.stra_get_secs())
                ids = context.stra_sell(self.__code__, 0, abs(curPos), "sell")  # 设置为0表示以市价卖出
                # 将订单号加入到管理中
                for localid in ids:
                    self.__orders__[localid] = localid
                # 更新入场时间
                self.__last_entry_time__ = now
            elif curPos < 0:  # 持有空头头寸，此时要进行多头平仓操作
                # 当前时间，一定要从api获取，不然回测会有问题
                now = makeTime(self.__ctx__.stra_get_date(), self.__ctx__.stra_get_time(),
                               self.__ctx__.stra_get_secs())
                ids = context.stra_buy(self.__code__, 0, abs(curPos), "buy")  # 设置为0表示以市价卖出
                # 将订单号加入到管理中
                for localid in ids:
                    self.__orders__[localid] = localid
                # 更新入场时间
                self.__last_entry_time__ = now
        # 进行每日平仓时，不生成信号
        return 0,0

    def make_deal(self, context: HftContext, signal, price,moves, mean_price_MA20, ATR):
        """
        根据信号执行交易
        """
        # 交易模块，以ATR作为入场信号,ATR越小越严格
        if signal != 0 and abs(price - mean_price_MA20) <= 1.15 * ATR:
            # 当前时间，一定要从api获取，不然回测会有问题
            now = makeTime(self.__ctx__.stra_get_date(), self.__ctx__.stra_get_time(),
                           self.__ctx__.stra_get_secs())
            if signal > 0:
                context.stra_log_text("出现正向信号")
                targetPx = price + 0.005 * self.__offset__
                ids = context.stra_buy(self.__code__, targetPx, moves, "buy")
                # 将订单号加入到管理中
                for localid in ids:
                    self.__orders__[localid] = localid
                # 更新入场时间
                self.__last_entry_time__ = now
            else:
                context.stra_log_text("出现反向信号")
                targetPx = price - 0.005 * self.__offset__
                ids = context.stra_sell(self.__code__, targetPx, moves, "sell")
                for localid in ids:
                    self.__orders__[localid] = localid
                # 更新入场时间
                self.__last_entry_time__ = now

    def normalized(self,df, name):
        # 提取"bid_ask_quantity"列
        nomal_data = df[name]

        # 创建一个标准化器对象
        scaler = StandardScaler()

        # 将数据进行标准化
        normal_sheet = scaler.fit_transform(nomal_data.values.reshape(-1, 1))
        return normal_sheet
    def probability(self):
        columns = ['time', 'exchg', 'code', 'price', 'open', 'high', 'low', 'settle_price', 'upper_limit',
                   'lower_limit', 'total_volume', 'volume', 'total_turnover', 'turn_over', 'open_interest',
                   'diff_interest', 'trading_date', 'action_date', 'action_time', 'pre_close', 'pre_settle',
                   'pre_interest', 'bid_price_0', 'bid_price_1', 'bid_price_2', 'bid_price_3', 'bid_price_4',
                   'bid_price_5', 'bid_price_6', 'bid_price_7', 'bid_price_8', 'bid_price_9', 'ask_price_0',
                   'ask_price_1', 'ask_price_2', 'ask_price_3', 'ask_price_4', 'ask_price_5', 'ask_price_6',
                   'ask_price_7', 'ask_price_8', 'ask_price_9', 'bid_qty_0', 'bid_qty_1', 'bid_qty_2', 'bid_qty_3',
                   'bid_qty_4', 'bid_qty_5', 'bid_qty_6', 'bid_qty_7', 'bid_qty_8', 'bid_qty_9', 'ask_qty_0',
                   'ask_qty_1', 'ask_qty_2', 'ask_qty_3', 'ask_qty_4', 'ask_qty_5', 'ask_qty_6', 'ask_qty_7',
                   'ask_qty_8', 'ask_qty_9']
        data = self.__last_tick__
        if data is not None:
            df = pd.DataFrame(list(zip(*[data]))).transpose()
            df.columns = columns
            df['price_change'] = df['pre_settle'] - df['open']
            df['price_range'] = df['high'] - df['low']
            df['open_interest_change'] = df['open_interest'] - df['pre_interest']
            df['price_settle_diff'] = df['pre_close'] - df['pre_settle']
            df['bid_ask_quantity'] = (df['ask_qty_0']+df['ask_qty_1']+df['ask_qty_2']+df['ask_qty_3']+df['ask_qty_4']+
                                      df['bid_qty_0']+df['bid_qty_1']+df['bid_qty_2']+df['bid_qty_3']+df['bid_qty_4'])
            df = df[['price', 'volume', 'price_change', 'price_range', 'open_interest_change', 'price_settle_diff', 'bid_ask_quantity']]

            # 选择需要计算相关系数的因子列
            # df['price_change_normal'] = self.normalized(df,'price_change')
            # df['price_range_normal'] = self.normalized(df,'price_range')
            # df['volume_normal'] = self.normalized(df,'volume')
            # df['bid_ask_quantity_normal'] = self.normalized(df,'bid_ask_quantity')
            # df['open_interest_change_normal'] = self.normalized(df,'open_interest_change')
            # df['price_settle_diff_normal'] = self.normalized(df,'price_settle_diff')
            # selected_factors = ['price_change_normal', 'price_range_normal', 'volume_normal', 'open_interest_change_normal', 'price_settle_diff_normal', 'bid_ask_quantity_normal']
            selected_factors = ['price_change', 'price_range', 'volume',
                                'open_interest_change', 'price_settle_diff', 'bid_ask_quantity']
            # 提取相关因子的数据
            data = df[selected_factors].values
            # 加载模型
            loaded_model = load_model('D:/best_model4.h5')
            # 使用模型进行预测
            predictions = loaded_model.predict(data.astype(float))
        return predictions

    def on_tick(self, context:HftContext, stdCode:str, newTick:dict):

        if self.__code__ != stdCode:
            return

        #如果有未完成订单，则进入订单管理逻辑
        if len(self.__orders__.keys()) != 0:
            self.check_orders()
            return

        if not self.__channel_ready__:
            return



        #如果已经入场，则做频率检查
        if self.__last_entry_time__ is not None:
            #当前时间，一定要从api获取，不然回测会有问题
            now = makeTime(self.__ctx__.stra_get_date(), self.__ctx__.stra_get_time(), self.__ctx__.stra_get_secs())
            span = now - self.__last_entry_time__
            if span.total_seconds() <= 30:
                return


        # 信号标志
        signal = 0
        # 读取品种属性，主要用于价格修正
        commInfo = context.stra_get_comminfo(self.__code__)
        #最新价
        price = newTick["price"]
        #获取当前切片的时间
        time_judge = int(str(newTick['time'])[8:12])

        # 以dataframe的形式保存每一切片的数据，用于计算均线和布林线，中午收盘的时间不算
        data = pd.DataFrame([[newTick['time'], newTick['price'], newTick['high'], newTick['low'], newTick['pre_close']]],
                            columns=['time', 'price', 'high', 'low', 'pre_close'])
        self.boll_data = self.boll_data.append(data, ignore_index=True)

        # 开盘后30分钟、午盘和尾盘15分钟波动太大，不做交易
        if (113000000 >= int(str(newTick['time'])[8::]) >= 103000000) or (133000000 <= int(str(newTick['time'])[8::]) <= 150000000):
            self.probability()
            # 获得最新价格切片的索引
            i = len(self.boll_data) - 1
            # 计算均线和标准差
            mean_price_MA20, std_data_MA20 = self.compute_boll(i, 30)
            ATR = self.ATR(i, 14)
            # 中心价格（波动）
            centre_price = mean_price_MA20
            # 网格间距（波动）
            gap = std_data_MA20 * 2
            #生成网格
            band = self.grid_with_gap(centre_price,gap)
            grid = pd.cut([price], band, labels=[1, 2, 3, 4, 5, 6, 7, 8])[0]
            curPos = context.stra_get_position(self.__code__)

            #止损模块
            if self.stop_loss(context, curPos, newTick, ATR):
                return

            #止盈模块
            if self.take_profit(context, curPos, newTick, ATR):
                return

            #出场模块
            if self.stop_trade(context,price, curPos, mean_price_MA20, ATR, time_judge):
                print(self.accumulated_tick)
                print('###########')
                print(time_judge)
                return

            #生成信号阶段
            ##每日平仓
            if self.day_close:
                #daily_close中包含了将参数重置的部分
                signal,moves = self.daily_close(context, curPos, time_judge)
                self.make_deal(context, signal, price,moves, mean_price_MA20, ATR)
                # 只要满足入场条件，将用于出场模块的累计切片数重置为0
                self.accumulated_tick = 0
            ##不每日平仓
            elif not self.day_close:
                signal, moves = self.signal_with_moves(context, grid)
                # 每日收盘更新格子的初始值
                if time_judge >= 1515:
                    self.last_grid = 0
                    self.grid_change_last = [0, 0]
                    self.last_deal = {}
                    self.__last_tick__ = None
                    self.boll_data = pd.DataFrame(None, columns=['time', 'price', 'high', 'low', 'pre_close'])
                    self.accumulated_tick = 0
                    return
                self.make_deal(context, signal, price,moves, mean_price_MA20, ATR)

                #只要满足入场条件，将用于出场模块的累计切片数重置为0
                self.accumulated_tick = 0

        #更新上一个切片的数据
        self.__last_tick__ = newTick


    def on_bar(self, context:HftContext, stdCode:str, period:str, newBar:dict):
        return

    def on_channel_ready(self, context:HftContext):
        undone = context.stra_get_undone(self.__code__)
        if undone != 0 and len(self.__orders__.keys()) == 0:
            context.stra_log_text("%s存在不在管理中的未完成单%f手，全部撤销" % (self.__code__, undone))
            isBuy = (undone > 0)
            ids = context.stra_cancel_all(self.__code__, isBuy)
            for localid in ids:
                self.__orders__[localid] = localid
            self.__cancel_cnt__ += len(ids)
            context.stra_log_text("cancelcnt -> %d" % (self.__cancel_cnt__))
        self.__channel_ready__ = True

    def on_channel_lost(self, context:HftContext):
        context.stra_log_text("交易通道连接丢失")
        self.__channel_ready__ = False

    def on_entrust(self, context:HftContext, localid:int, stdCode:str, bSucc:bool, msg:str, userTag:str):
        if bSucc:
            context.stra_log_text("%s下单成功，本地单号：%d" % (stdCode, localid))
        else:
            context.stra_log_text("%s下单失败，本地单号：%d，错误信息：%s" % (stdCode, localid, msg))

    def on_order(self, context:HftContext, localid:int, stdCode:str, isBuy:bool, totalQty:float, leftQty:float, price:float, isCanceled:bool, userTag:str):
        if localid not in self.__orders__:
            return

        if isCanceled or leftQty == 0:
            self.__orders__.pop(localid)
            if self.__cancel_cnt__ > 0:
                self.__cancel_cnt__ -= 1
                self.__ctx__.stra_log_text("cancelcount -> %d" % (self.__cancel_cnt__))
        return

    def on_trade(self, context:HftContext, localid:int, stdCode:str, isBuy:bool, qty:float, price:float, userTag:str):
        self.last_deal = {'localid': localid, 'stdCode': stdCode, 'isBuy': isBuy, 'qty': qty, 'price': price,
                          'userTag': userTag}
        return
