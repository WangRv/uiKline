# encoding:UTF-8
import datetime as dt

datetime_format = "%Y-%m-%d%H:%M:%S"
def control_trade_time(now_time: dt.datetime) -> bool:
    weekday_variable = [day for day in range(0,5)] # 5,6 aren't working day
    datetime_str = now_time.strftime("%Y-%m-%d")

    morning_opening_time = dt.datetime.strptime(datetime_str+"09:30:00",datetime_format)
    morning_closing_time = dt.datetime.strptime(datetime_str+"11:30:00",datetime_format)

    afternoon_opening_time = dt.datetime.strptime(datetime_str+"13:00:00",datetime_format)
    afternoon_closing_time = dt.datetime.strptime(datetime_str+"15:00:00",datetime_format)
    now_weekday = now_time.weekday()
    if not now_weekday in weekday_variable:
        return False
    elif (morning_opening_time<=now_time<morning_closing_time) or \
        (afternoon_opening_time<=now_time<afternoon_closing_time):
        return True
    else:
        return False
if __name__ == '__main__':
    # test time function
    now = dt.datetime.now()
    print(control_trade_time(now))