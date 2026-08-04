from datetime import datetime
import random

def get_current_time():
    return {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

def get_weather(city:str):
    temperatures={"Ottawa":25,"Toronto":27,"Montreal":24,"Vancouver":21}
    return {"city":city,"temperature":temperatures.get(city,random.randint(18,30)),"condition":"Sunny"}

def multiply(a:float,b:float):
    return {"result":a*b}
