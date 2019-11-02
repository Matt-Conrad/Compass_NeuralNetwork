# A Compass Should Always Point North

In the CompassImages folder you will see a collection of 16 images. There is an arrow and the letter "N" in each image. The arrow points either up, right, down, or left. Additionally, the placement of "N" is also in those same directions. This image set contains all the possible combinations of the configurations. I wanted to build a neural network that would properly classify images where the arrow was correctly pointing to the "N". The goal for this project was to get familiar with building a neural network in Tensorflow. I used a manually created feature matrix for easy visualization and a basic 2-layer neural network that has been previously used to classify the XOR problem. This project also serves as a starting point for using more complex computer vision techniques to create features. 

### Dependencies

I'm using the following software in this project:
* Python 3.7.3 
* Matplotlib 3.1.0
* Numpy 1.16.4
* Pandas 0.24.2
* Scikit-Learn 0.21.2
* Tensorflow 1.14.0

### Results

See "CompassProblemPresentation.pptx" for in-depth results and visuals. Most notably, I was able to get 7/8 test points guessed correctly with the neural network I trained on 8 training points with a tensorflow-made neural network. 
