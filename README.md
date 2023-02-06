# National AI Student Challenge 2022
Category B (with Coding)
### Team: Noshinom

## How to run our code
1. Ensure that you have PeekingDuck installed. `pip install peekingduck`
2. Run `peekingduck run` to run our main code.

## Purpose
Our tool allows us to help businesses do fruit and vegetable inspection more efficiently, by differentiating edible and inedible fruits and vegetables.

We will be focusing on the ***PeekingDuck*** AI Brick.

## Data and model
Dataset from: https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset

We specifically used the mango data in this dataset. This is so we can curate our model towards a specific fruit, to test accuracy.

You can find all data used in the [model_data](model_data) folder.

We trained the model with images of **fresh** and **rotten** mango, allowing it to learn to differentiate the two. During training, the data in [mangoData](model_data/mangoData/) was split further into training and validation sets, and then tested against images in [mangoTest](model_data/mangoTest/).

Our average accuracy was about 93% after training. 

## PeekingDuck
Followed the custom node tutorial here: https://peekingduck.readthedocs.io/en/stable/tutorials/06_using_your_own_models.html#model-training

In the folder [custom_nodes](src/custom_nodes), we have the [fruit_classifier.py](src/custom_nodes/model/fruit_classifier.py), and its manifest file [fruit_classifier.yml](src/custom_nodes/configs/model/fruit_classifier.yml).

With our class labels as {0: 'fresh', 1: 'rotten'}, we are able to run the pipeline in [pipeline_config.yml](pipeline_config.yml) to output a CSV of the predictions of images in the [inspection](model_data/inspection) folder.

### Accuracy
Our average accuracy was about 95%.

## Further Improvements
We hope to be able to use PeekingDuck and other resources to have our input as ***live feed from the camera***, instead of images from a folder. 

Additionally, we hope to design a model that is able to differentiate edibility for all kinds of fruits, using not just visual imagery, but infrared, chemical, etc.