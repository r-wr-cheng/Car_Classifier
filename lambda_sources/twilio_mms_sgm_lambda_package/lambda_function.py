import urllib3
from PIL import Image
import numpy as np
import boto3
import io
import os

def lambda_handler(event, context):
    
    from_number = event['fromNumber']
    pic_url = event['image']
    num_media = event['numMedia']

    if num_media != '0':
        http = urllib3.PoolManager()
        response = http.request('GET', pic_url)
        
        scaled_image = Image.open(io.BytesIO(response.data)).convert('L').resize((200,200))
        
        payload = ''
        for i in np.asarray(scaled_image.convert('L')).flatten():
            payload += f'{i},'
        payload = payload[:-1]
        
        client = boto3.client('runtime.sagemaker')
        response = client.invoke_endpoint(EndpointName=os.environ.get('SGM_ARN'),
                                       ContentType='text/csv',
                                       Body=payload)
        
        prediction = response['Body'].read().decode()
        
        twilio_resp = prediction

    else:
        twilio_resp = 'Something fucked up'
    
    return '<?xml version=\"1.0\" encoding=\"UTF-8\"?>'\
           f'<Response><Message><Body>{twilio_resp}</Body></Message></Response>'
