from wtpy import BaseHftStrategy
from wtpy import HftContext

from datetime import datetime

def float_range(start, end, step):
    current = start
    while current <= end:
        yield current
        current = round(current + step,3)

def makeTime(date:int, time:int, secs:int):
    '''
    将系统时间转成datetime\n
    @date   日期，格式如20200723\n
    @time   时间，精确到分，格式如0935\n
    @secs   秒数，精确到毫秒，格式如37500
    '''
    return datetime(year=int(date/10000), month=int(date%10000/100), day=date%100, 
        hour=int(time/100), minute=time%100, second=int(secs/1000), microsecond=secs%1000*1000)

class OrderFlow(BaseHftStrategy):

    def __init__(self, name:str, code:str, expsecs:int, offset:int,tickcnt:int, N:float, M: int, F:float, freq:int=30, inday = False):
        BaseHftStrategy.__init__(self, name)

        '''交易参数'''
        self.tickcnt = tickcnt          #回调tick的数量
        self.__code__ = code            #交易合约
        self.__expsecs__ = expsecs      #订单超时秒数
        self.__offset__ = offset        #指令价格偏移
        self.__freq__ = freq            #交易频率控制，指定时间内限制信号数，单位秒
        self.N = N                      #失衡倍数
        self.M = M                      #堆积条件数
        self.F = F                      #惩罚参数
        self.inday = inday              #是否在日内平仓
        self.cleartimes = [[1459,1515]] #清盘时间
        self.stable = 120              #累计亏损时间上限

 
        '''内部数据'''
        self.tcnt = 0                   #经历tick的个数（tickcnt为一个周期）
        self.signal = 0                 #信号指数
        self.__last_tick__ = None       #上一笔行情
        self.__orders__ = dict()        #策略相关的订单
        self.__last_entry_time__ = None #上次入场时间
        self.__cancel_cnt__ = 0         #正在撤销的订单数
        self.__channel_ready__ = False  #通道是否就绪
        self.act_buy={}                 #主动买入
        self.act_sell={}                #主动卖出
        self.count = 0                  #累计亏损时间计数
        self.max = 0                    #最大盈利的价格
        self.profit = False             #每次交易是否经历盈利区间
        self.volumesum = 0              #计算每个分析单元的成交订单和
        self.last_deal = {}             #记录上一笔成交的信息
        

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
            if span.total_seconds() > self.__expsecs__: #如果订单超时，则需要撤单
                for localid in self.__orders__:
                    self.__ctx__.stra_cancel(localid)
                    self.__cancel_cnt__ += 1
                    self.__ctx__.stra_log_text("cancelcount -> %d" % (self.__cancel_cnt__))

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
    
    def count_order(self, context:HftContext, stdCode:str, newTick:dict): #估计成交量模块（主动买入以及主动卖出）
        b_p = newTick["bid_price_0"]
        a_p = newTick["ask_price_0"]
        b_q = newTick["bid_qty_0"]
        a_q = newTick["ask_qty_0"]
        l_b_p = self.__last_tick__["bid_price_0"]
        l_a_p = self.__last_tick__["ask_price_0"]
        l_b_q = self.__last_tick__["bid_qty_0"]
        l_a_q = self.__last_tick__["ask_qty_0"]
        
        if b_p==l_b_p:
            if b_q<l_b_q:
                self.act_sell[b_p]=self.act_sell.get(b_p,0)+l_b_q-b_q
        if b_p< l_b_p:
            for i in range(1,5):
                if b_p < self.__last_tick__["bid_price_" + str(i)]:
                    self.act_sell[self.__last_tick__["bid_price_" + str(i)]]=self.act_sell.get(self.__last_tick__["bid_price_" + str(i)],0) + self.__last_tick__["bid_qty_" + str(i)]
                elif b_p==self.__last_tick__["bid_price_" + str(i)]:
                    if self.__last_tick__["bid_qty_" + str(i)]>b_q:
                        self.act_sell[self.__last_tick__["bid_price_" + str(i)]]=self.act_sell.get(self.__last_tick__["bid_price_" + str(i)],0) + self.__last_tick__["bid_qty_" + str(i)]-b_q
                        break
                else:
                    break

        if a_p==l_a_p:
            if a_q<l_a_q:
                self.act_buy[a_p]=self.act_buy.get(a_p,0)+l_a_q-a_q
        if a_p>l_a_p:
            for i in range(1,5):
                if a_p > self.__last_tick__["ask_price_" + str(i)]:
                    self.act_buy[self.__last_tick__["ask_price_" + str(i)]]=self.act_buy.get(self.__last_tick__["ask_price_" + str(i)],0) + self.__last_tick__["ask_qty_" + str(i)]
                elif a_p==self.__last_tick__["ask_price_" + str(i)]:
                    if self.__last_tick__["ask_qty_" + str(i)]>a_q:
                        self.act_buy[self.__last_tick__["ask_price_" + str(i)]]=self.act_buy.get(self.__last_tick__["ask_price_" + str(i)],0) + self.__last_tick__["ask_qty_" + str(i)]-a_q
                        break
                else:
                    break
        
        self.__last_tick__ = newTick
        self.tcnt+=1
        self.volumesum+=newTick['volume']

    def inday_close(self, context:HftContext, stdCode:str, newTick:dict):#日内平仓模块
        curPos = context.stra_get_position(self.__code__)
        curTime = context.stra_get_time()
        for tmPair in self.cleartimes:
            if curTime >= tmPair[0] and curTime <= tmPair[1]:
                if curPos !=0:
                    now = makeTime(self.__ctx__.stra_get_date(), self.__ctx__.stra_get_time(), self.__ctx__.stra_get_secs())
                    if curPos > 0:
                        ids = context.stra_sell(self.__code__, 0, 1, "sell")
                        for localid in ids:
                            self.__orders__[localid] = localid
                        self.__last_entry_time__ = now

                    if curPos < 0:
                        ids = context.stra_buy(self.__code__, 0, 1, "buy")
                        for localid in ids:
                            self.__orders__[localid] = localid
                        self.__last_entry_time__ = now
                self.standardize("day_close",newTick)
                return True
            return False

    def stop_loss(self, context:HftContext, stdCode:str, newTick:dict):#止损模块
        price = newTick['price']
        curPos = context.stra_get_position(self.__code__)
        if curPos !=0:
            now = makeTime(self.__ctx__.stra_get_date(), self.__ctx__.stra_get_time(), self.__ctx__.stra_get_secs())
            if curPos>0:
                if self.last_deal['price']-price > 0.01:
                    self.count+=1
                    if self.count>=self.stable:
                        ids = context.stra_sell(self.__code__, 0, 1, "sell")
                        for localid in ids:
                            self.__orders__[localid] = localid
                        self.__last_entry_time__ = now
                        self.standardize("in_day",newTick)
                        return True
                else:
                    self.count = 0
                    return False
            if curPos<0:
                if price-self.last_deal['price'] > 0.01:
                    self.count+=1
                    if self.count>=self.stable:
                        ids = context.stra_buy(self.__code__, 0, 1, "buy")
                        for localid in ids:
                            self.__orders__[localid] = localid
                        self.standardize("in_day",newTick)
                        return True
                else:
                    self.count = 0
                    return False
                
    def take_profit(self, context:HftContext, stdCode:str, newTick:dict):#止盈模块
        price = newTick['price']
        curPos = context.stra_get_position(self.__code__)
        if curPos !=0:
            now = makeTime(self.__ctx__.stra_get_date(), self.__ctx__.stra_get_time(), self.__ctx__.stra_get_secs())
            if curPos>0:
                if price>self.last_deal['price']:
                    if self.profit == False:
                        self.max = price
                    self.profit = True
                if price> self.max:
                    self.max = price
                if self.profit:
                    if self.max-self.last_deal['price']>0.05:
                        if self.max-price>0.4*(self.max-self.last_deal['price']):
                            ids = context.stra_sell(self.__code__, 0, 1, "sell")
                            for localid in ids:
                                self.__orders__[localid] = localid
                            self.__last_entry_time__ = now
                            self.standardize("in_day",newTick)
                            return True
                    else:
                        if self.max-price>0.1:
                            ids = context.stra_sell(self.__code__, 0, 1, "sell")
                            for localid in ids:
                                self.__orders__[localid] = localid
                            self.__last_entry_time__ = now
                            self.standardize("in_day",newTick)
                            return True

            if curPos<0:
                if price<self.last_deal['price']:
                    if self.profit == False:
                        self.max = price
                    self.profit = True
                if price< self.max:
                    self.max = price
                if self.profit:
                    if self.last_deal['price']-self.max>0.05:
                        if price-self.max>0.4*(self.last_deal['price']-self.max):
                            ids = context.stra_buy(self.__code__, 0, 1, "buy")
                            for localid in ids:
                                self.__orders__[localid] = localid
                            self.__last_entry_time__ = now
                            self.standardize("in_day",newTick)
                            return True
                    else:
                        if price-self.max>0.1:
                            ids = context.stra_buy(self.__code__, 0, 1, "buy")
                            for localid in ids:
                                self.__orders__[localid] = localid
                            self.__last_entry_time__ = now
                            self.standardize("in_day",newTick)
                            return True

    def signial_gen(self, context:HftContext, stdCode:str, newTick:dict):
        a_b=[]
        a_s=[]
        sum_b=0
        sum_s=0
        de_ac=False
        su_ac=False
        for k,v in self.act_buy.items():
            sum_b+=v
            a_b.append(k)
        for k,v in self.act_sell.items():
            sum_s+=v
            a_s.append(k)
        if sum_b+sum_s<0.8*self.volumesum:
            self.volumesum = 0 
            self.act_buy={}
            self.act_sell={}
            self.tcnt=0
            return
        if sum_b+sum_s>3*self.volumesum:
            self.volumesum = 0 
            self.act_buy={}
            self.act_sell={}
            self.tcnt=0
            return
        c=0
        if len(a_b)==0:
            pass
        else:
            for i in float_range(min(a_b),max(a_b),0.005):
                if self.act_buy.get(i,0)>=self.N*self.act_sell.get(round(i-0.005,3),1):
                    c+=1
                    if c>=self.M:
                        de_ac = True
                else:
                    c=0
        c=0
        if len(a_s)==0:
            pass
        else:
            for i in float_range(min(a_s),max(a_s),0.005):
                if self.act_sell.get(i,0)>=self.N*self.act_buy.get(round(i+0.005,3),1):
                    c+=1
                    if c>=self.M:
                        su_ac = True
                else:
                    c=0
        if de_ac==True and su_ac==False:
            if self.signal>=0:
                self.signal+=1
            if self.signal<0:
                self.signal+=self.F
        if de_ac==False and su_ac==True:
            if self.signal<=0:
                self.signal-=1
            if self.signal>0:
                self.signal-=self.F
        '''if de_ac==True and su_ac==True:
            if sum_b>sum_s:
                if self.signal>=0:
                    self.signal+=1
                if self.signal<0:
                    self.signal+=self.F
            if sum_b<sum_s:
                if self.signal<=0:
                    self.signal-=1
                if self.signal>0:
                    self.signal-=self.F
        '''

    
    def stretagy_make(self, context:HftContext, stdCode:str, newTick:dict):
        price = newTick['price']
        if self.signal != 0:
                curPos = context.stra_get_position(self.__code__)
                now = makeTime(self.__ctx__.stra_get_date(), self.__ctx__.stra_get_time(), self.__ctx__.stra_get_secs())
                if self.signal > 0 and curPos < 0:
                    ids = context.stra_buy(self.__code__, 0, 1, "buy")
                    for localid in ids:
                        self.__orders__[localid] = localid           
                    self.__last_entry_time__ = now
                    self.signal = 0
                elif self.signal > 1 and curPos == 0:
                    ids = context.stra_buy(self.__code__, round(price+0.005,3), 1, "buy")
                    for localid in ids:
                        self.__orders__[localid] = localid
                    self.__last_entry_time__ = now
                    self.profit = False
                elif self.signal < 0 and curPos > 0:
                    ids = context.stra_sell(self.__code__, 0, 1, "sell")
                    for localid in ids:
                        self.__orders__[localid] = localid
                    self.__last_entry_time__ = now
                    self.signal = 0
                elif self.signal < -1 and curPos == 0:
                    ids = context.stra_sell(self.__code__, round(price-0.005,3), 1, "sell")
                    for localid in ids:
                        self.__orders__[localid] = localid
                    self.__last_entry_time__ = now
                    self.profit = False
    
    def standardize(self,condition,newTick):
        if condition=="day_close":
            self.signal=0
            self.act_buy={}
            self.act_sell={}
            self.__last_tick__=None
            self.tcnt=0
            self.count=0
            self.profit=False
            self.volumesum = 0
            self.max = 0
        if condition=="in_day":
            self.signal = 0
            self.act_buy={}
            self.act_sell={}
            self.tcnt=0
            self.count=0
            self.profit=False
            self.__last_tick__ = newTick
            self.volumesum=0
            self.max=0

    def on_tick(self, context:HftContext, stdCode:str, newTick:dict):
        if self.__code__ != stdCode:
            return

        if len(self.__orders__.keys()) != 0:
            self.check_orders()
            return

        if not self.__channel_ready__:
            return

        if self.__last_tick__ == None:
            self.__last_tick__ = newTick
            return 
        

        #日内平仓模块
        if self.inday:
            if self.inday_close(context, stdCode, newTick):
                return

        #止盈模块
        if self.take_profit(context, stdCode, newTick):
            return
        
        #止损模块
        if self.stop_loss(context, stdCode, newTick):
            return
        
        #订单量估计模块              
        self.count_order(context, stdCode, newTick)

        if self.tcnt==self.tickcnt:
            #信号产生模块
            self.signial_gen(context, stdCode, newTick)
            #策略执行模块
            self.stretagy_make(context, stdCode, newTick)
        
            self.volumesum = 0 
            self.act_buy={}
            self.act_sell={}
            self.tcnt=0