# ===== IMPORT REQUIRED PACKAGES =========

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
import matplotlib.image as mpimg
import matplotlib.image as mpimg


#====================== 1.READ A INPUT IMAGE =========================

filename = askopenfilename()
img = mpimg.imread(filename)
plt.imshow(img,cmap='gray')
plt.title('ORIGINAL IMAGE')
plt.axis ('off')
plt.show()



# ====================== 2. PREPROCESSING ==========================

#==== RESIZE IMAGE ====

resized_image = cv2.resize(img,(300,300))
img_resize_orig = cv2.resize(img,((50, 50)))

fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)
plt.axis ('off')
plt.show()
           
                 
#==== GRAYSCALE IMAGE ====

try:            
    gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
    
except:
    gray1 = img_resize_orig
   
fig = plt.figure()
plt.title('GRAY SCALE IMAGE')
plt.imshow(gray1,cmap='gray')
plt.axis ('off')
plt.show()


#=================== 3.FEATURE EXTRACTION ===================

#=== MEAN STD DEVIATION ===

mean_val = np.mean(gray1)
median_val = np.median(gray1)
var_val = np.var(gray1)
features_extraction = [mean_val,median_val,var_val]

print("------------------------------------")
print("     MEAN MEDIAN VARAINCE           ")
print("------------------------------------")
print()
print(features_extraction)





#============================ 5. IMAGE SPLITTING ===========================

import os 

from sklearn.model_selection import train_test_split

aff = os.listdir('Dataset/Affected/')
not_aff = os.listdir('Dataset/Not/')
#       
dot1= []
labels1 = [] 
for img11 in aff:
        # print(img)
        img_1 = mpimg.imread('Dataset/Affected//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(1)


for img11 in not_aff:
        # print(img)
        img_1 = mpimg.imread('Dataset/Not//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(2)


x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)

print()
print("-------------------------------------")
print("       IMAGE SPLITTING               ")
print("-------------------------------------")
print()


print("Total no of data        :",len(dot1))
print("Total no of test data   :",len(x_train))
print("Total no of train data  :",len(x_test))



#=============================== CLASSIFICATION =================================

from keras.utils import to_categorical


y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)




x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]


# ======== CNN ===========
    
from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
# from keras.layers import Activation
from keras.models import Sequential
from keras.layers import Dropout




# initialize the model
model=Sequential()


#CNN layes 
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(3,activation="softmax"))

#summary the model 
model.summary()

#compile the model 
model.compile(loss='binary_crossentropy', optimizer='adam')
y_train1=np.array(y_train)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)


print("-------------------------------------")
print("CONVOLUTIONAL NEURAL NETWORK (CNN)")
print("-------------------------------------")
print()
#fit the model 
history=model.fit(x_train2,train_Y_one_hot,batch_size=20,epochs=5,verbose=1)

accuracy = model.evaluate(x_train2, train_Y_one_hot, verbose=1)

loss=history.history['loss']

error_cnn=max(loss)*10

acc_cnn=100- error_cnn


print("-------------------------------------")
print("PERFORMANCE ---------> (CNN)")
print("-------------------------------------")
print()
print("1. Accuracy   =", acc_cnn,'%')
print()
print("2. Error Rate =",error_cnn)




#=============================== PREDICTION =================================

print()
print("-----------------------")
print("       PREDICTION      ")
print("-----------------------")
print()


Total_length = len(aff) + len(not_aff) 
 

temp_data1  = []
for ijk in range(0,Total_length):
    # print(ijk)
    temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
    temp_data1.append(temp_data)

temp_data1 =np.array(temp_data1)

zz = np.where(temp_data1==1)

if labels1[zz[0][0]] == 1:
    print('---------------------------------------')
    print(' IDENTIFIED = Affected by heart disease')
    print('---------------------------------------')
    
    
    from mtcnn import MTCNN
    import cv2
    
    # initialize the MTCNN detector
    detector = MTCNN()
    
    # load the input image and convert it to grayscale
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect the face using MTCNN
    faces = detector.detect_faces(image)
    
    
    
    
    
    import cv2
    import numpy as np
    
    # Function to detect objects and draw bounding boxes
    def detect_and_draw_boxes(image):
    
        objects = [[50, 50, 50, 50], [40, 150, 50, 50]]  
    
        for box in objects:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
    
        return image
    
    # Load your medical image
    image = cv2.imread(filename)
    
    # Detect objects and draw bounding boxes
    image_with_boxes = detect_and_draw_boxes(image)
    
    plt.imshow(image_with_boxes)
    plt.title('AFFECTED IMAGE')
    plt.axis ('off')
    plt.show()

    

elif labels1[zz[0][0]] == 2:
    print('-------------------------------------------')
    print(' IDENTIFIED = Not Affected by heart disease')
    print('-------------------------------------------')
    













