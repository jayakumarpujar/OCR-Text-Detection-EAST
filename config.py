@jayakumar

import yaml

with open('hyperparameter.yaml') as file:
    hyperparameters = yaml.load(file, Loader = yaml.FullLoader)

training_data_path = "/training/"
checkpoint_path = "/models/"
pretrained_model_path = "/base_model/"+hyperparameters['pretrained_checkpoint_name']
total_steps = int(hyperparameters['total_steps'])
save_checkpoint_steps = int(hyperparameters['save_checkpoint_steps'])
max_checkpoints_to_keep = int(hyperparameters['max_checkpoints_to_keep'])
batch_size_per_gpu = int(hyperparameters['max_checkpoints_to_keep'])
learning_rate = float(hyperparameters['learning_rate'])
num_readers = int(hyperparameters['num_readers'])
geometry = hyperparameters['geometry']
restore = bool(hyperparameters['restore'])
output_model_path = hyperparameters['output_model']


