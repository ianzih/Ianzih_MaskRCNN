# Ianzih_MaskRCNN

This code from [matterport_Mask_RCNN](https://github.com/matterport/Mask_RCNN)

I made some revise.

## environment
1. use tensorflow-gpu==1.14.0
2. scikit-image==0.16.2
3. numpy==1.16.0
4. Keras==2.2.0

this is my DockerHub : (https://hub.docker.com/repository/docker/ianzih/maskrcnn_tensorflow-gpu)

## Start
```bash
mkdir -p ./logs   #save your pre-trained models
```

## Testing Visualize Mask
```bash
python test_tongue.py
```

## training
1. Prepare 
    ```bash
   mkdir -p ./tongue_dataset   #save your tongue dataset
    ```
2. traing
    ```bash
    python tongue.py
    ```
