# OCR Text-Detection-EAST
OpenCV Text Detection (EAST text detector) OCR


# TD Training

Create an environment with python version 3.7.0 and create conda environment 
Install the requirements using pip install -r requirements.txt
Don't install any other dependencies

Change these variables in config.py script.
     training_data_path : dataset folder path
     checkpoint_path : output checkpoints folder path
     pretrained_model_path  : pretrained checkpoints folder path
     
Pretrained model checkpoints can be used if available otherwise use resnet_v1.ckpt addded in the repository
If required change the other hyperparameters accordingly.
      
Command to run the multigpu_train.py is
       python3 multigpu_train.py
       
To check loss of all the models
tensorboard --logdir='./checkpoint_foldername/

find the model which has very low loss by running the above command 
then convert the best model into pb 

# To convert model into pb make changes in config_freeze.py 
     
    checkpoint_file : output checkpoint folder path ( give the checkpoint number which you want to convert )
    output_graph_name : output pb model path ( Including the desired model output name) 
     
Command to run the script freeze_graph-new.py 
       python3 freeze_graph-new.py
       
This pb model can be used for further testing.
