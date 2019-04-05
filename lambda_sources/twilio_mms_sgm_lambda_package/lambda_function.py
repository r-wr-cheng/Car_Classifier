import urllib3

def lambda_handler(event, context):
    
    from_number = event['fromNumber']
    pic_url = event['image']
    num_media = event['numMedia']

    if num_media != '0':
        http = urllib3.PoolManager()
        r = http.request('GET', pic_url)
        twilio_resp = 'Picture Recieved!'

    else:
        twilio_resp = 'Something fucked up'
    
    return twilio_resp