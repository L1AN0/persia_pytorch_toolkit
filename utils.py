import datetime

def current_time_str():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
