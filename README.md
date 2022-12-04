# CARLA Work
This repository is about CARLA research at UW-Madison

## First Glance
CARLA project is an open-source driving project designed in a simulated environment. One advantage of using CARLA is that you could test the models in a relatively costless way. We performed object detection works in the CARLA simulator and would like to share our results.

## DETR
The article "End-to-End Object Detection with Transformers" introduced us with DETR model, it uses a CNN backbone and a transformer to achieve the end-to-end object detection goal. It outputs the bounding boxes, probabilities, and categories directly without needing of post-processing such as non-max suppression (a bipartite matching helped in this), anchor boxes designing, etc. We chose to use this model and run tests on it.

## KITTI
The KITTI dataset is a famous dataset for autonomous driving field. We choose to use KITTI 2-D object detection dataset with images and labels. We also convert the KITTI dataset to COCO format to better train it on the DETR model.

## Run Original Pre-trained DETR on CARLA
We are trying to integrate DETR model into the CARLA simulator. DETR is an end-to-end detection model with great capability, we consider it would be a good object detection model for detecting vehicle flow and monitor traffic. To do that, we need to implement the model into the CARLA simulator.

### Demo Results
![image](https://user-images.githubusercontent.com/68500948/205504700-e87f7879-31b1-4f16-ab92-b8f48d9c97b6.png)
![image](https://user-images.githubusercontent.com/68500948/205504705-0303d931-ad7a-4181-a5bd-b9e2bce54c08.png)
![image](https://user-images.githubusercontent.com/68500948/205504707-9a2896de-80f3-4102-810c-d24d1ef75e10.png)


## Transfer Learning on DETR with KITTI
We would like to explore more on KITTI and DETR. Specifically, we would like to fit DETR to specialize in autonomous driving scenarios. We chose to do transfer learning on KITTI with DETR res-50 version and removed its fc layer weights. We used the official training script from Facebook research DETR repo. 
![image](https://user-images.githubusercontent.com/68500948/205504836-e0f4d084-65db-4c76-9ccc-2037ca092217.png)

### 4-epoch Result on KITTI Test Set Images
![image](https://user-images.githubusercontent.com/68500948/205504920-7d698abe-920d-4757-98b9-a47f2f7f58ea.png)
![image](https://user-images.githubusercontent.com/68500948/205504929-2f223e15-ba9b-422f-86dd-262b08bc8030.png)
![image](https://user-images.githubusercontent.com/68500948/205504950-659d7b9f-5d06-4072-a1ab-872728b27493.png)

We can observe that the transfer learning result is not bad in the test set of KITTI. However, there are some problems. For example, some predictions' bounding boxes appear not to be extremely accurate.

E.g.,
![image](https://user-images.githubusercontent.com/68500948/205505029-601028cd-22c3-4869-9ed9-d59367fd0f6a.png)

Reasons could be we only trained limited epochs despite we are using transfer learning. One way to find out is to train more epoch (probably 10) and compare the result with the current model. Unfortunately, we do not have such a huge computational power and we recommend future researcher to explore more on that.

### 4-epoch Result on CARLA Simulator
![image](https://user-images.githubusercontent.com/68500948/205505274-efdf60d9-a2ce-4c6f-89df-fb6f1e4c1a25.png)
![image](https://user-images.githubusercontent.com/68500948/205505360-92f7775a-04e9-4bb5-b350-c86b74d1da26.png)

The result appear to be neither good or bad. We do discovered some problems.

#### Confidence Threshold
If the Confidence Threshold is too large, many targets would be missed out. For example<br>
![image](https://user-images.githubusercontent.com/68500948/205505213-bc8b1c9a-40a7-4025-b055-357f6e46040f.png)<br>
You can see that there is only one prediction but rather multiple objects.

We also cannot adjust the confidence to a very small value. In that case, there would be many unwanted boxes which could compromise our result. We discovered that setting the thresold to near 0.5 is a good choice.

#### Simulated Result, Size, and Transformation
Other problems include that CARLA is a simulator, the image from virtual camera still have some differences to the real data. Specifically, the KITTI datasets' images appear to be more clear and high-resolutional, the one from CARLA is more blury, more synthesize-look. That could be one of the reason why our transfer learned DETR performs better in real datasets. Another conjecture would be the size of the objects and angle. In the KITTI dataset, the camera is at the front and take picture at a flat angle, in the result above we attached the camera to a high position and look down. To determine if that is a factor, we conducted another demo using camera attach to a vehicle.

![image](https://user-images.githubusercontent.com/68500948/205506902-525c4445-0f36-41c6-be3c-d17183890906.png)
![image](https://user-images.githubusercontent.com/68500948/205506916-cb5594b5-562c-4b5f-9e80-e1cba9fb569f.png)

The results are better than the results from a looking-down camera view.

Furthermore, we consider doing appropriate transformation on images would improve the quality of prediction. We leave this to further researchers.

### Discussion

### Summary

### Other Notes
We deleted the .pth in folder weights and output, the full weight dicts are so large so that GitHub does not allow. Contact me (williamdlye(at)outlook) if you need these.

### References
https://github.com/thedeepreader/detr_tutorial
https://github.com/facebookresearch/detr
