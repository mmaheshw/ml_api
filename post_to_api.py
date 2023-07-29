"""
Script to post to FastAPI instance for model inference
author: Manjari Maheshwari
Date: July 2023
"""

import requests
import json

#url = "enter render web app url here"
url = "https://project-api-2cbe.onrender.com"


# explicit the sample to perform inference on
sample =  { 'age':50,
            'workclass':"Private",  
            'fnlgt':234721,
            'education':"Doctorate",
            'education_num':16,
            'marital_status':"Separated",
            'occupation':"Exec-managerial",
            'relationship':"Not-in-family",
            'race':"Black",
            'sex':"Female",
            'capital_gain':0,
            'capital_loss':0,
            'hours_per_week':50,
            'native_country':"United-States"
            }

data = json.dumps(sample)

headers = {'Content-Type': 'application/json'}

# post to API and collect response
response = requests.post(url, data=data, headers=headers )

# display output - response will show sample details + model prediction added
print("response status code", response.status_code)
print("response content:")
# Print the raw response before trying to parse it as JSON
print("Raw response:", response.text)

# Try to parse the response as JSON and handle the exception if it fails
try:
    print(response.json())
except json.decoder.JSONDecodeError:
    print("Invalid JSON response")








