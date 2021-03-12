Install the following packages into your environment first. Run them in terminal  
!python -m pip install 'fsspec>=0.3.3'   
!python -m pip install dask[bag] --upgrade  

Run the following line to preprocess the dataset. Enter the paths to the folders in ' '  
!python "utils/preprocess.py" 5 'SOURCE_FOLDER_CONTAINING_UNZIPPED_DATASET' 'DESTINATION_FOLDER'  

Before running, check the trainSRGAN.py file and change all the paths accordingly.  
All the .py files necessary for modules in trainSRGAN.py have been uploaded to utils.
