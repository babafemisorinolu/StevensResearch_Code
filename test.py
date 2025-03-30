
import requests
from time import sleep
#include json library
import json
import ast
url='http://127.0.0.1:8001/audio/result/90ae6196-0b07-49c4-81e6-0995fba0bf29'

resp = requests.get(url=url).json()
status=resp['status']
if(status=="Processing"):
#sleep(30)
    sleep(120)

R=(resp['prediction'])
# print('response',resp)
# print(R)
# print(type(R))
# R= '{"first_name": "Michael", "last_name": "Rodgers", "department": "Marketing"}'

#convert string to  object
# json_object = json.loads(R)

#check new data type
# print(type(json_object))

res = ast.literal_eval(R)
# print(res)
# print(type(res))
pred_dict={}
for key, value in res.items():
    if(isinstance(value,tuple)):
        if value[0] in pred_dict:
            #check if the max value is in this key
            if(pred_dict[value[0]]<value[1]):
                pred_dict[value[0]]=value[1] #swap with the max
        else:
            pred_dict[value[0]]=value[1] #insert the key, value
        # pred_dict[key]=value[0]
        # print(key, '->', value[0], ' type:', type(value))
    else:
        # print(key, '->', value) #classification output alone
        pass

# print(R)
print(pred_dict)
# pred_dict["Heater"]=9
# pred_dict["AC"]=9
# pred_dict["TV"]=9

msg="sending a notification to the residents phone to turn off the "
devices="";
for key, value in pred_dict.items():
    if (len(devices)== 0):
        devices=key
    else:
        devices = devices + " and " + key;

noti_msg=msg+devices
print(noti_msg)