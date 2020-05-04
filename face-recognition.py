
from keras.models import load_model
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
import mtcnn
from numpy import savez_compressed
from numpy import load
from numpy import expand_dims

# load the facenet model
model = load_model('facenet_keras.h5')
print(model.inputs)
print(model.outputs)

# Function for extracting a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

# load images and extract faces for all images in a directory
def load_faces(directory):
    faces = list()
    for filename in listdir(directory):
       path = directory + filename
       face = extract_face(path)
       faces.append(face)
       return faces
   

# load a dataset that contains one subdir for each celebrity which contains images
def load_dataset(directory):
	X, y = list(), list()
	for subdir in listdir(directory):
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)

# load train dataset
trainX, trainy = load_dataset('My-celeb-pics/training_set/')
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset('My-celeb-pics/test_set/')
print(testX.shape, testy.shape)
# save arrays to one file in compressed format
savez_compressed('3-bolywood-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample (expanding dimensions)
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]


# load the face dataset
data = load('3-bolywood-celebrity-faces-dataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# load the facenet model
model = load_model('facenet_keras.h5')

# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
    embeddings = get_embedding(model, face_pixels)
    newTrainX.append(embeddings)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)

# convert each face in the test set to an embedding
newTestX = list()
for face_pixels in testX:
    embeddings = get_embedding(model, face_pixels)
    newTestX.append(embeddings)
newTestX = asarray(newTestX)
print(newTestX.shape)

# save arrays to one file in compressed format
savez_compressed('3-bolywood-celebrity-faces-embeddings.npz', newTrainX, trainy, newTestX, testy)

# develop a classifier for the 3 Celebrity Faces Dataset

from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC


data = load('3-bolywood-celebrity-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

in_encoder = Normalizer(norm = 'l2')
out_encoder = LabelEncoder()
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
trainy = out_encoder.fit_transform(trainy)
testy = out_encoder.transform(testy)

classifier = SVC(kernel = 'linear', probability = True)
classifier.fit(trainX, trainy)

# predict
yhat_train = classifier.predict(trainX)
yhat_test = classifier.predict(testX)
# score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)

print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

from random import choice

testX_faces = data['arr_2']

# test model on a random example from the test dataset
selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])
# prediction for the face
samples = expand_dims(random_face_emb, axis=0)
yhat_class = classifier.predict(samples)

# get name
class_index = yhat_class[0]
predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s' % predict_names[0])
print('Expected: %s' % random_face_name[0])


    
    
    
        
     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        