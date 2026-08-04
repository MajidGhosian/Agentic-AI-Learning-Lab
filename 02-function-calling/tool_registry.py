from tools import get_current_time,get_weather,multiply

TOOLS={
"get_current_time":{"function":get_current_time,"schema":{"type":"function","function":{"name":"get_current_time","description":"Returns the current local time.","parameters":{"type":"object","properties":{}}}}},
"get_weather":{"function":get_weather,"schema":{"type":"function","function":{"name":"get_weather","description":"Get weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}},
"multiply":{"function":multiply,"schema":{"type":"function","function":{"name":"multiply","description":"Multiply two numbers.","parameters":{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}}}}
}
