from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model
import pickle
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from tkinter import simpledialog
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

main = tkinter.Tk()
main.title("Movies Recommendation using CNN & ANN") #designing main screen
main.geometry("1300x1200")

global filename
global dataset
global movies
global X_train, X_test, y_train, y_test
global X, Y
global cnn_model
global train, test

def uploadDataset():
    global dataset
    global movies
    text.delete('1.0', END)
    #filename = filedialog.askdirectory(initialdir=".")commented
    #text.insert(END,filename+" loaded\n\n")commented
    dataset = pd.read_csv('Dataset/ratings.csv', nrows=10000,sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating'])
    movies = pd.read_csv('Dataset/movies.csv',encoding='latin-1',sep='\t')
    text.insert(END,str(dataset))
    text.insert(END,str(movies))

def preprocessDataset():
    global X_train, X_test, y_train, y_test
    global X, Y
    text.delete('1.0', END)
    temp = dataset.values
    X = temp[:,0:2]
    Y = temp[:,2]
    text.insert(END,X)#HAS COLUMNS OF USER_ID AND MOVIE_ID
    text.insert(END,Y)#HAS ID_RATING
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)    
    text.insert(END,"Movies Train & Test Model Generated\n\n")
    text.insert(END,"Total Dataset Size : "+str(len(X))+"\n")
    text.insert(END,"Splitted Training Size : "+str(len(X_train))+"\n")
    text.insert(END,"Splitted Test Size : "+str(len(X_test))+"\n\n\n")            

def runLR():
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    predict = lr.predict(X_test) 
    lr_loss = 1 - accuracy_score(y_test,predict)
    text.insert(END,"Logistic Regression Error Loss/Score : "+str(lr_loss)+"\n")

def runRF():
    global X_train, X_test, y_train, y_test
    #text.delete('1.0', END)
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    predict = rf.predict(X_test) 
    rf_loss = 1 - accuracy_score(y_test,predict)
    text.insert(END,"Random Forest Error Loss/Score : "+str(rf_loss)+"\n")    

def runANN():
    global dataset
    global train, test
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)
    n_users = len(dataset.user_id)
    movie_id = len(dataset.movie_id)
    movie_input = Input(shape=[1], name="Movie-Input")
    movie_embedding = Embedding(movie_id+1, 5, name="ANNMovie-Embedding")(movie_input)
    movie_vec = Flatten(name="Flatten-Movies")(movie_embedding)

    user_input = Input(shape=[1], name="User-Input")
    user_embedding = Embedding(n_users+1, 5, name="ANNUser-Embedding")(user_input)
    user_vec = Flatten(name="Flatten-Users")(user_embedding)

    prod = Dot(name="Dot-Product", axes=1)([movie_vec, user_vec])
    ann_model = Model([user_input, movie_input], prod)
    ann_model.compile('adam', 'mean_squared_error')
    print(ann_model.summary())
    if os.path.exists('model/annmodel.h5'):
        ann_model = load_model('model/annmodel.h5')
        f = open('model/annhistory.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        loss = data['loss']
        text.insert(END,"ANN Error Loss/Score : "+str(loss[9])+"\n")
    else:
        history = ann_model.fit([train.user_id, train.movie_id], train.rating, epochs=10, verbose=1)
        ann_model.save('model/annmodel.h5')
        f = open('model/annhistory.pckl', 'wb')
        pickle.dump(history.history, f)
        f.close()
        loss = history.history['loss']
        text.insert(END,"ANN Error Loss/Score : "+str(loss[9])+"\n")

def runCNN():
    global dataset
    global cnn_model
    global train, test
    n_users = len(dataset.user_id)
    movie_id = len(dataset.movie_id)
    movie_input = Input(shape=[1], name="Movie-Input")
    movie_embedding = Embedding(movie_id+1, 5, name="CNNMovie-Embedding")(movie_input)
    movie_vec = Flatten(name="Flatten-Movies")(movie_embedding)
    user_input = Input(shape=[1], name="User-Input")
    user_embedding = Embedding(n_users+1, 5, name="CNNUser-Embedding")(user_input)
    user_vec = Flatten(name="Flatten-Users")(user_embedding)
    # CNN layer concatenate features
    conc = Concatenate()([movie_vec, user_vec])
    # add CNN fully-connected-layers
    fc1 = Dense(128, activation='relu')(conc)
    fc2 = Dense(32, activation='relu')(fc1)
    out = Dense(1)(fc2)
    # Create model and compile it
    cnn_model = Model([user_input, movie_input], out)
    cnn_model.compile('adam', 'mean_squared_error')
    print(cnn_model.summary())
    if os.path.exists('model/cnnmodel.h5'):
        cnn_model = load_model('model/cnnmodel.h5')
        f = open('model/cnnhistory.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        loss = data['loss']
        text.insert(END,"CNN Error Loss/Score : "+str(loss[9])+"\n")
    else:
        history = cnn_model.fit([train.user_id, train.movie_id], train.rating, epochs=10, verbose=1)
        cnn_model.save('model/cnnmodel.h5')
        f = open('model/cnnhistory.pckl', 'wb')
        pickle.dump(history.history, f)
        f.close()
        loss = history.history['loss']
        text.insert(END,"CNN Error Loss/Score : "+str(loss[9])+"\n")
    
def graph():
    f = open('model/annhistory.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    ann_loss = data['loss']
    
    f = open('model/cnnhistory.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    cnn_loss = data['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(ann_loss, 'ro-', color = 'red')
    plt.plot(cnn_loss, 'ro-', color = 'green')
    plt.legend(['ANN Loss', 'CNN Loss'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('Epoch/Iteration Wise Loss Graph')
    plt.show()

def recommendation():
    global dataset
    text.delete('1.0', END)
    userid = simpledialog.askinteger(title="Input User ID",prompt="Enter User ID for Movies Recommendation",parent=main)
    movie_data = np.array(list(set(dataset.movie_id)))
    user = np.array([1 for i in range(len(movie_data))])
    predictions = cnn_model.predict([user, movie_data])
    predictions = np.array([a[0] for a in predictions])
    recommended_movie_ids = (-predictions).argsort()[:len(dataset)]
    print(recommended_movie_ids)
    output = movies[movies['movie_id'].isin(recommended_movie_ids)]
    output = output.values
    index = 0
    #flag = 1
    for i in range(len(output)):
        #if output[i,0] == userid and flag == 0:
        #    flag = 1
        #   text.insert(END,'User ID : '+str(userid)+" Recommended Movie : "+str(output[i,2])+" Movie Type : "+str(output[i,3])+"\n\n")
        #if flag == 1 and index < 10:
        if index<10:
            text.insert(END,'User ID : '+str(userid)+" Recommended Movie : "+str(output[i,2])+" Movie Type : "+str(output[i,3])+"\n\n")
            index = index + 1
        elif index > 10:
            break
            
        

    
font = ('times', 16, 'bold')
title = Label(main, text='Movies Recommendation using CNN & ANN')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Movies Dataset", command=uploadDataset, bg='#ffb3fe')
uploadButton.place(x=840,y=100)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=preprocessDataset, bg='#ffb3fe')
processButton.place(x=840,y=150)
processButton.config(font=font1) 

rfButton = Button(main, text="Run Logistic Regression Algorithm", command=runLR, bg='#ffb3fe')
rfButton.place(x=840,y=200)
rfButton.config(font=font1) 

rfButton = Button(main, text="Run Random Forest Algorithm", command=runRF, bg='#ffb3fe')
rfButton.place(x=840,y=250)
rfButton.config(font=font1) 

annButton = Button(main, text="Run ANN Algorithm", command=runANN, bg='#ffb3fe')
annButton.place(x=840,y=300)
annButton.config(font=font1)

cnnButton = Button(main, text="Run CNN Algorithm", command=runCNN, bg='#ffb3fe')
cnnButton.place(x=840,y=350)
cnnButton.config(font=font1)

graphButton = Button(main, text="ANN & CNN Error Loss Graph", command=graph, bg='#ffb3fe')
graphButton.place(x=840,y=400)
graphButton.config(font=font1)

predictButton = Button(main, text="Recommend Movie", command=recommendation, bg='#ffb3fe')
predictButton.place(x=840,y=450)
predictButton.config(font=font1)

main.config(bg='RoyalBlue2')
main.mainloop()
