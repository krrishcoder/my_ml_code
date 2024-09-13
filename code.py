import pandas as pd
import numpy as np
import easyocr
import cv2
import numpy as np
import requests
import pickle
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("/kaggle/input/note03data/train.csv")

reader = easyocr.Reader(['en'])

def get_image_for_url(url):
    # Fetch the image data
    response = requests.get(url)
    image_data = np.asarray(bytearray(response.content), dtype=np.uint8)

    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    
    return img

def print_text_only(resultt):
    str = ""
    for (bbox , text, prob) in resultt :
        str = str +" "+ text
        
    return str


data['text_extracted'] = 'default v'


for i in range(20000 , 30000):
    url = data.iloc[i].image_link
    arr = get_image_for_url(url)
    
    result = ""
    try :
        result = reader.readtext(arr)
    except e :
        continue
    
    str = print_text_only(result)
    data['text_extracted'][i] = str
    print(i)


    

