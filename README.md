# Applied ML Project
## Summary
Our task was to build a machine learning algorithim that could determine if a square was in an image. We decided to use a linear support vector machine as our classifier, that ingested feature vectors extracted using the histogram of gradients algorithim (also known as H.O.G). 

Examples of our test data set taken from (https://data.mendeley.com/datasets/wzr2yv7r53/1):

![square-1](https://user-images.githubusercontent.com/33464965/100675587-14714300-331c-11eb-96fa-f0a4ec683745.png)
![square-10](https://user-images.githubusercontent.com/33464965/100675891-a9743c00-331c-11eb-9214-94fb089d39f6.png)
![square-16](https://user-images.githubusercontent.com/33464965/100675929-ba24b200-331c-11eb-82ad-3128e4616f3f.png)
![square-3](https://user-images.githubusercontent.com/33464965/100675962-c872ce00-331c-11eb-9ec3-df897ad8ec3b.png)

Since the square we were trying to classifiy varied in shape, location, and rotation we had to incorporate additonal techniques such as a sliding window and the image pyramid in order for our histogram of gradient algorithim to correctly extract the feature vectors from the image. 

The result of our classifier yielded a 91% success rate. We noticed that the classifer failed to classifiy very small squares correctly. This is due to the image pyramid scaling down the image (zooming out) which works to fit squares that are larger than the sliding window, but not smaller squares. In order to correct this we would need to introduce an image pyramid that also scales the image up, such that smaller squares fill up a larger portion of the sliding window, and more features are extracted.

Example histogram of gradients, prediction label, actual label, and bounding box selected by the algorithim:

![hog_5](https://user-images.githubusercontent.com/33464965/100677743-85b2f500-3320-11eb-8b8a-89bde1230b9d.jpg)
![hog_6](https://user-images.githubusercontent.com/33464965/100677839-ba26b100-3320-11eb-8217-441df5fef3e6.jpg)
![hog_0](https://user-images.githubusercontent.com/33464965/100677718-77fd6f80-3320-11eb-9623-31aa75664f6d.jpg)
![pred_5](https://user-images.githubusercontent.com/33464965/100677731-8055aa80-3320-11eb-9f54-f00ede28b407.jpg)
![pred_6](https://user-images.githubusercontent.com/33464965/100676618-1b995080-331e-11eb-85ee-04478f78ecbb.jpg)
![pred_2](https://user-images.githubusercontent.com/33464965/100677704-73d15200-3320-11eb-8c2d-4461c968f411.jpg)

## How to Run
Install Python 3 if you have not already (https://www.python.org/downloads/). 

When you are ready run the following commands in the terminal at the root folder containing the project.

### Install Dependencies
Install Python packages (only need to run once): `pip install -r requirements.txt`

### Python File
1. `python square_detection.py`.

### Jupyter Notebook
1. Run this line in your terminal at the root folder: `jupyter notebook`. It will open up a window in your browser. Look for `square_detection.ipynb` file and click on it.
