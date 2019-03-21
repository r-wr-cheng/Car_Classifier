#aws s3 cp --recursive s3://car-classifier-us-east-2/python_readable_data/stanford_cars_dataset/convnet_model/ ~/convnet_model/

aws s3 cp s3://car-classifier-us-east-2/python_readable_data/stanford_cars_dataset/scaled_bounded_grayscale_dict.p ~ 

aws s3 cp s3://car-classifier-us-east-2/python_readable_data/stanford_cars_dataset/stanford_labels_cleaned.csv ~

aws s3 cp s3://car-classifier-us-east-2/python_readable_data/stanford_cars_dataset/training_data_meta.csv ~
