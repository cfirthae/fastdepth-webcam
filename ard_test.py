import requests

thingIP = "192.168.31.124"

arduinoIP = "http://192.168.31.124/lgd/199"

try:
	requests.get(url = arduinoIP)

except:
	pass

arduinoIP = "http://" + thingIP + "/lgd/919"

try:
	requests.get(url = arduinoIP)

except:
	pass

arduinoIP = "http://" + thingIP + "/lgd/991"

try:
	requests.get(url = arduinoIP)

except:
	pass


