# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud, STOPWORDS
from geopy.geocoders import Nominatim
import folium
import plotly.graph_objs as go
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, GlobalMaxPool1D, Dropout, Conv1D, Activation, Input, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
import tensorflow as tf


df = pd.read_csv('Datafiniti_Fast_Food_Restaurants_May19.csv')

#lets take a look 
df

df.shape

#checking dfthe info 
df.info()

#checking null values
df.isna().sum()

# Handling missing values in 'websites' column
df['websites'].fillna('Unknown', inplace=True)

#looking into city variable
df['city'].value_counts()[:20].plot(kind= 'bar')

df['name'].value_counts()[:20].plot(kind= 'bar')

df['new_categories']=df['categories'].apply(lambda x:x.lower())

df[['categories','new_categories']].head(5)

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

df['new_categories'] = df['categories'].str.lower().apply(
    lambda x: " ".join(word for word in x.split() if word not in stop_words)
)

df[['categories','new_categories']].head(5)

# Remove unwanted text patterns from 'new_categories' efficiently
unwanted_terms = [
    'restaurants', 'restaurant', 'take out', 'carry-out food', 'manufacturers',
    'cypress station', 'caterers', 'delivery service', 'uncategorized', 
    "women's clothing", 'delis', 'hotels and motel', '&', 'catering', 
    'airport devonshire', 'airport', 'bars', '- full service', 'clubs', 
    'pubs', 'american canoga park', 'american cape fear', 
    'american downtown blacksburg'
]

for term in unwanted_terms:
    df['new_categories'] = df['new_categories'].str.replace(term, '', regex=True)


df[['categories','new_categories']].head(5)

# Creating a summary DataFrame for number of unique values in each column
unique_df = pd.DataFrame()
unique_df['Features'] = df.columns
unique_df['Uniques'] = [df[col].nunique() for col in df.columns]

# Plotting the unique values
plt.figure(figsize=(15, 7))
sns.barplot(x=unique_df['Features'], y=unique_df['Uniques'], alpha=1)
plt.title('Bar Plot for Number of Unique Values in Each Column', weight='bold', size=15)
plt.ylabel('#Unique values', size=12, weight='bold')
plt.xlabel('Features', size=12, weight='bold')
plt.xticks(rotation=90)
plt.show()


all_words = ' '.join([text for text in df['new_categories']])
wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=110).generate(all_words)

plt.figure(figsize=(15, 15))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Categories", fontsize=24)
plt.show()


#for the count of rest. area wise we groupby addres and count of rest. reat_name
df1 = df.groupby(['province'])['name'].aggregate('count').reset_index().sort_values('name', ascending=False)
df1.rename(columns = {'name':'Number_of_resto'}, inplace = True)

#info of df1
df1.info()

#displaying the values in df1
df1.head(10)

#ploting the graph between restaurnt name and addres
fig , ax = plt.subplots(figsize = (18,8))
splot = sns.barplot(x = 'province', y = 'Number_of_resto' , data = df1.head(10) , ax=ax)

# now simply assign the bar values to
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center',
                   va = 'center', xytext = (0, 9), textcoords = 'offset points')

#for the count of rest. city wise we groupby city and count of rest. reat_name
df1 = df.groupby(['city'])['name'].aggregate('count').reset_index().sort_values('name', ascending=False)
df1.rename(columns = {'name':'Number_of_resto'}, inplace = True)

#ploting the graph between restaurnt name and city
fig , ax = plt.subplots(figsize = (18,8))
splot = sns.barplot(x = 'city', y = 'Number_of_resto' , data = df1.head(10) , ax=ax)

# now simply assign the bar values to
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center',
                   va = 'center', xytext = (0, 9), textcoords = 'offset points')

from datetime import timedelta
#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

import requests # library to handle requests
from pandas import json_normalize  # new line

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

address = 'US'

geolocator = Nominatim(user_agent="tr_explorer", timeout=10)  # Set timeout to 10 seconds
try:
    location = geolocator.geocode(address)
    if location:
        latitude = location.latitude
        longitude = location.longitude
        print('The geographical coordinate of US are {}, {}.'.format(latitude, longitude))
    else:
        print('Location not found.')
except Exception as e:
    print(f"Error occurred: {e}")


# create map of Toronto using latitude and longitude values
map_bng = folium.Map(location=[latitude, longitude], zoom_start=9)

# add markers to map
for latitude, longitude, address, city in zip(df['latitude'], df['longitude'], df['address'], df['city']):
    label = '{}, {}'.format(address, city)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [latitude, longitude],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_bng)  
    
map_bng

df.fillna('', inplace=True)
df["detailed_address"] = df["name"] + ' ' + df["address"] + ' ' + df["city"] + ' ' + df["country"] + ' ' + df["latitude"].astype(str) + ' ' + df["longitude"].astype(str) + ' ' + df["postalCode"] + ' ' + df['province'] + ' ' + df['websites']

df1 = df[["detailed_address","new_categories"]]

new_df = pd.concat([df1.drop('new_categories', axis=1), df['new_categories'].str.get_dummies(sep=",")], axis=1)
print(new_df)

new_df

new_df.columns

#to see columns converting it into list
columns = new_df.columns.tolist()

'''#printing all colm names 
for x in range(len(columns)):
    print (columns[x])'''

# dropping unwanted variables
new_df1 =new_df.drop([ 'lounge','mediterranean' , 'office' ,'australian','auto leasing','auto renting','auto repairing','automated teller machines (atm)',
                      'autos','baby gear','baby products','bagels','bagels','bakeries',"bakers' supplies",'bakery','bakery' ,'bar supplies','barbecue grills supplies',
                      'barboursville ','bartending service','bath products','beauty salons','bedding','beverage','beverages','beverages retail','big box store',
                      'billiard parlors','billiard table ','biosil','birmingham ', 'bistro','bistro', 'bistros','black mountain ', 'bladensburg ', 'bliss',
                      'blytheville ', 'bothell ', 'bowling','box lunches','braun','brew ', 'brewers','brookneal ', 'brooksville ', 'broomall ' ,'brunch' ,'buffet' 
                      ,'buffet'  ,'building contractors', 'business  personal coaches','business development','business schools','business services','cabinets',
                      'cable internet','cable tv','caf','caf diplomat','cafes','cafeteria ', 'cafeterias','canby ','canonsburg ','cantonese ','carnivals','carpenters',
                      'carry out','cary ','casino','cedar hill ','cedar rapids ','centre ','centreville ', 'chain','chain ','charleston ','charlotte ','cheap eats',
                      'check cashing service','chelsea ','child care','childersburg ',"children's clothing",'clarisonic','clayton ','clermont ','clothing',
                      'clothing alterations','cocktail bar','cocoa ','coffee brewing devices','coffee makers','coffee retail','coffee tea','coffeehouses',
                      'collectibles','college academic building','college quad','columbus ','comfort food ','commercial photographers','commercial printing',
                      'commercial refrigerators','communications','computer internet services','computer online services','concessionaires','construction storage',
                      'consumer electronics','copy centers','copying  duplicating services','corolla ','cosmetics','crab house ','craft supplies','craig ', 
                      'creole  cajun ', 'creole cajun ' ,'crestview acres','crooksville ' ,'crosstown plaza','crystal river ','cumberland furnace ','cuyahoga falls ' 
                      ,'dayton','dayton','delivery','deordorant','delicatessen','delicatessen', 'des moines ','dialysis','dialysis clinics','diesel fuel','discover',
                      'dive ','doctor','dog run','dothan ','dvd','e commerce','e-commerce','eastern european ', 'eating','educational materials',
                      'educational service-business','electric contractors','electronic publishing','employment opportunities','english ' ,
                      'engravers','entertainment  arts','essential oils','ethnic food markets','ethnic markets','european','event planners','event planning',
                      'event ticket agencies','evergreen ' ,'exporters','face cleansers','family ' ,'family entertainment','family-friendly dining','farming service',
                      'farmington ' ,'farms','filipino','finance  financial services','financial planning','fine dining ','fish  seafood markets','fish  seafood retail',
                      'fish market','fishing tackle','florists','food  beverage s','food  dining','food  entertainment','food court','food dining','food drink',
                      'food east columbus','food products','food s','food service','food service management','food truck','foods','forever living','fort pierce ',
                      'franchising','fund raising games','fur repair','furniture','fusion','fusion ','gadsden ','gainesville ', 'garden centers','garment services',
                      'gas station','gay lesbian ','general contractors','general entertainment','general merchandise-retail','geologists','german ','gillette',
                      'gladstone ', 'gluten-free','gluten-free','global','glue','gluten-free foods','golf courses','golf practice ranges','gourmet ', 'grand island ',
                      'grants pass ','greek  park 100','green products','grocers-ethnic foods','hair care products','hand sanitizer','hawaiian', 'headquarters',
                      'health care providers','health clinics', 'health food ','health food store','health medical services','healthy','hermiston ', 'hilo ',
                      'historic sites','historic walking areas','holding companies','home furnishings','home services  furnishings','hospitals','hotel bar',
                      'hunan ', 'huntersville','huntersville', 'huntington ', ], axis=1)

new_df1.info()

bar_plot = pd.DataFrame()
bar_plot['cat'] = new_df1.columns[1:]
bar_plot['count'] = new_df1.iloc[:,1:].sum().values
bar_plot.sort_values(['count'], inplace=True, ascending=False)
bar_plot.reset_index(inplace=True, drop=True)
bar_plot.head()

#taking the cusines that count more than 50
main_categories = bar_plot[bar_plot['count']>50]
categories = main_categories['cat'].values
categories = np.append(categories,'Others')
not_category = []
new_df1['Others'] = 0

for i in new_df1.columns[1:]:
    if i not in categories:
        new_df1.loc[new_df1[i] == 1, 'Others'] = 1
        not_category.append(i)

new_df1.drop(not_category, axis=1, inplace=True)

new_df1 = new_df1.dropna()

main_categories

bar_plot1 = pd.DataFrame()
bar_plot1['cat'] = new_df1.columns[1:]
categories1= bar_plot1['cat'].values
categories1 = np.append(categories1,'Others')
categories1

most_common_cat = pd.DataFrame()
most_common_cat['cat'] = new_df1.columns[1:]
most_common_cat['count'] = new_df1.iloc[:,1:].sum().values
most_common_cat.sort_values(['count'], inplace=True, ascending=False)
most_common_cat.reset_index(inplace=True, drop=True)
most_common_cat.head()

# new_df1 = new_df1.dropna()

rowSums = new_df1.iloc[:,1:].sum(axis=1)
multiLabel_counts = rowSums.value_counts()

boxplot = new_df1.copy()
boxplot['len'] = new_df1.detailed_address.apply(lambda x: len(x))

from wordcloud import WordCloud,STOPWORDS

plt.figure(figsize=(15,15))
text = new_df1.detailed_address.values
cloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))
plt.axis('off')
plt.title("Common words on the description",fontsize=40)
plt.imshow(cloud)

# Create a DataFrame to store results for each model across different metrics
results = pd.DataFrame(columns=['Model', 'AUC', 'Precision', 'Recall', 'F1'])


#
seeds = [1, 43, 678, 90, 135]

t = results.copy()
t

X_train, X_test, y_train, y_test = train_test_split(new_df1['detailed_address'], 
                                                    new_df1[new_df1.columns[1:]], 
                                                    test_size=0.3, 
                                                    random_state=seeds[4], 
                                                    shuffle=True)
vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
vectorizer.fit(X_train)

X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

# Add this function before the evaluation loops
def train_and_evaluate_model(model_pipeline, model_name, X_train, y_train, X_test, y_test):
    auc = precision = recall = f1 = 0
    total_categories = len(y_train.columns)

    for category in y_train.columns:
        print(f'**Processing {category} titles...**')

        model_pipeline.fit(X_train, y_train[category])
        prediction = model_pipeline.predict(X_test)

        auc += roc_auc_score(y_test[category], prediction)
        precision += precision_score(y_test[category], prediction, zero_division=1)
        recall += recall_score(y_test[category], prediction, zero_division=1)
        f1 += f1_score(y_test[category], prediction, zero_division=1)

    average_auc = auc / total_categories
    average_precision = precision / total_categories
    average_recall = recall / total_categories
    average_f1 = f1 / total_categories

    print(f'Average AUC ROC is {average_auc}')
    print(f'Average Precision is {average_precision}')
    print(f'Average Recall is {average_recall}')
    print(f'Average F1 Score is {average_f1}')

    return {
        'Model': model_name,
        'AUC': average_auc,
        'Precision': average_precision,
        'Recall': average_recall,
        'F1': average_f1
    }


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier


# Logistic Regression
LR_pipeline = Pipeline([('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1))])
new_result = train_and_evaluate_model(LR_pipeline, 'Logistic Regression (OneVsAll)', X_train, y_train, X_test, y_test)
results = pd.concat([results, pd.DataFrame([new_result])], ignore_index=True)




from sklearn.naive_bayes import MultinomialNB

# Naive Bayes
NB_pipeline = Pipeline([('clf', OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None)))])
new_result = train_and_evaluate_model(NB_pipeline, 'Naive Bayes', X_train, y_train, X_test, y_test)
results = pd.concat([results, pd.DataFrame([new_result])], ignore_index=True)



from sklearn.svm import LinearSVC

# SVM
SVC_pipeline = Pipeline([('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1))])
new_result = train_and_evaluate_model(SVC_pipeline, 'SVM', X_train, y_train, X_test, y_test)
results = pd.concat([results, pd.DataFrame([new_result])], ignore_index=True)


from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier



# Random Forest
RF_pipeline = Pipeline([('clf', OneVsRestClassifier(RandomForestClassifier(), n_jobs=-1))])
new_result = train_and_evaluate_model(RF_pipeline, 'Random Forest', X_train, y_train, X_test, y_test)
results = pd.concat([results, pd.DataFrame([new_result])], ignore_index=True)






results

#this will take so much time so run for only once and save the results
'''from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB

classifier = BinaryRelev(GausancesianNB())
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
accuracy_score(y_test,predictions)
print('AUC ROC is {}'.format(roc_auc_score(y_test,predictions.toarray())))'''

#results.loc[4,'BinaryRelevance'] = roc_auc_score(y_test,predictions.toarray())
#results

#this will take so much time so run for only once and save the results
'''from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression

classifier = ClassifierChain(LogisticRegression())
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

print('AUC ROC is {}'.format(roc_auc_score(y_test,predictions.toarray())))'''

#results.loc[4,'ClassifierChain'] = roc_auc_score(y_test,predictions.toarray())
#results

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

# KNN (Multiple Output)
clf = MultiOutputClassifier(KNeighborsClassifier()).fit(X_train, y_train)
predictions = clf.predict(X_test)

auc = roc_auc_score(y_test, predictions, average='macro')
precision = precision_score(y_test, predictions, average='macro', zero_division=1)
recall = recall_score(y_test, predictions, average='macro', zero_division=1)
f1 = f1_score(y_test, predictions, average='macro', zero_division=1)

print(f'AUC ROC: {auc}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

knn_result = {
    'Model': 'MultipleOutput (KNN)',
    'AUC': auc,
    'Precision': precision,
    'Recall': recall,
    'F1': f1
}
results = pd.concat([results, pd.DataFrame([knn_result])], ignore_index=True)





# Helper function to evaluate deep learning model predictions
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    # Convert predictions to binary values (threshold at 0.5)
    predictions_binary = (predictions > 0.5).astype(int)

    auc = roc_auc_score(y_test, predictions_binary, average='macro')
    precision = precision_score(y_test, predictions_binary, average='macro', zero_division=1)
    recall = recall_score(y_test, predictions_binary, average='macro', zero_division=1)
    f1 = f1_score(y_test, predictions_binary, average='macro', zero_division=1)

    print(f'AUC ROC: {auc}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    return auc, precision, recall, f1


new_df1['detailed_address']

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=5000, lower=True)
tokenizer.fit_on_texts(new_df1['detailed_address'])
sequences = tokenizer.texts_to_sequences(new_df1['detailed_address'])
x = pad_sequences(sequences, maxlen=200)

X_train, X_test, y_train, y_test = train_test_split(x, 
                                                    new_df1[new_df1.columns[1:]], 
                                                    test_size=0.3, 
                                                    random_state=seeds[4])

# Remove duplicate class_weight calculation and generalize
most_common_cat['class_weight'] = len(most_common_cat) / most_common_cat['count']

# Generalized class weight calculation by converting it to a dictionary
class_weight = most_common_cat.set_index('cat')['class_weight'].to_dict()


# Display the updated DataFrame
most_common_cat.head()

num_classes = y_train.shape[1]
max_words = len(tokenizer.word_index) + 1
maxlen = 200

from tensorflow.keras.optimizers import Adam # - Works

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalMaxPool1D, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall

# Define the model
model = Sequential()

# Updated Embedding layer (Removed input_length argument as it is deprecated)
model.add(Embedding(input_dim=max_words, output_dim=20))

# Adding a dropout layer (uncomment if you need it)
# model.add(Dropout(0.2))

# GlobalMaxPool1D and final Dense layer
model.add(GlobalMaxPool1D())
model.add(Dense(num_classes, activation='sigmoid'))

# Compile the model


model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=[tf.keras.metrics.AUC(), Precision(), Recall()])


# Callbacks
callbacks = [
    ReduceLROnPlateau(),
    # Uncomment if you need early stopping
    # EarlyStopping(patience=10),
    ModelCheckpoint(filepath='model-simple.keras', save_best_only=True)  # Use `.keras` file extension as per the new standard
]

# Fit the model
history = model.fit(
    X_train, 
    y_train,
    class_weight=class_weight,
    epochs=30,
    batch_size=32,
    validation_split=0.3,
    callbacks=callbacks
)


dnn_model = model
# Metrics evaluation for DNN Model
metrics = evaluate_model(dnn_model, X_test, y_test)
dnn_result = {
    'Model': 'DNN',
    'AUC': metrics[0],
    'Precision': metrics[1],
    'Recall': metrics[2],
    'F1': metrics[3]
}
results = pd.concat([results, pd.DataFrame([dnn_result])], ignore_index=True)





from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding, Flatten, GlobalMaxPool1D, Dropout, Conv1D
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

filter_length = 300

# Define the model
model = Sequential()

# Updated Embedding layer (Removed input_length argument)
model.add(Embedding(input_dim=max_words, output_dim=20))

# Optionally add dropout
# model.add(Dropout(0.5))

# Add 1D Convolution layer
model.add(Conv1D(filters=filter_length, kernel_size=3, padding='valid', activation='relu', strides=1))

# Add pooling and output layers
model.add(GlobalMaxPool1D())
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

# Compile the model

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=[tf.keras.metrics.AUC(), Precision(), Recall()])

# Define callbacks
callbacks = [
    ReduceLROnPlateau(),
    ModelCheckpoint(filepath='model-conv1d.keras', save_best_only=True)  # Updated to use `.keras` extension
]

# Fit the model
history = model.fit(
    X_train,
    y_train,
    class_weight=class_weight,
    epochs=30,
    batch_size=32,
    validation_split=0.3,
    callbacks=callbacks
)


cnn_model = model
# Metrics evaluation for CNN Model
metrics = evaluate_model(cnn_model, X_test, y_test)

cnn_result = {
    'Model': 'CNN',
    'AUC': metrics[0],
    'Precision': metrics[1],
    'Recall': metrics[2],
    'F1': metrics[3]
}

# Use pd.concat to add the new result to results DataFrame
results = pd.concat([results, pd.DataFrame([cnn_result])], ignore_index=True)


from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()

glove_file = open('glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((max_words, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

from tensorflow.keras.layers import Input, Flatten, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf

# Input layer
deep_inputs = Input(shape=(maxlen,))

# Embedding layer using pre-trained weights (trainable=False)
embedding_layer = Embedding(input_dim=max_words, output_dim=100, weights=[embedding_matrix], trainable=False)(deep_inputs)

# LSTM layer
LSTM_Layer_1 = LSTM(128)(embedding_layer)

# Dense output layer
dense_layer_1 = Dense(99, activation='sigmoid')(LSTM_Layer_1)

# Creating the model
model = Model(inputs=deep_inputs, outputs=dense_layer_1)

# Callbacks - Updated ModelCheckpoint to use `.keras` extension
callbacks = [
    ReduceLROnPlateau(),
    ModelCheckpoint(filepath='model-lstm.keras', save_best_only=True)  # Updated the file extension to `.keras`
]

# Compiling the model
from tensorflow.keras.metrics import Precision, Recall

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=[tf.keras.metrics.AUC(), Precision(), Recall()])


# Fitting the model
history = model.fit(
    X_train, 
    y_train.values,
    class_weight=class_weight,
    batch_size=32,
    epochs=30,
    validation_split=0.3,
    callbacks=callbacks
)


# Metrics evaluation for LSTM Model
lstm_model = model
# Metrics evaluation for LSTM Model
metrics = evaluate_model(lstm_model, X_test, y_test)

lstm_result = {
    'Model': 'LSTM',
    'AUC': metrics[0],
    'Precision': metrics[1],
    'Recall': metrics[2],
    'F1': metrics[3]
}

# Use pd.concat to add the new result to results DataFrame
results = pd.concat([results, pd.DataFrame([lstm_result])], ignore_index=True)


# # Feature Scaling because yes we don't want one independent variable dominating the other and it makes computations easy
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# sequential model to initialise our ann and dense module to build the layers
from keras.models import Sequential
from keras.layers import Dense
#Initialising ANN
ann = tf.keras.models.Sequential()
 #Adding First Hidden Layer
ann.add(tf.keras.layers.Dense(units=1,activation="relu"))
 #Adding Second Hidden Layer
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
#Adding Output Layer
ann.add(tf.keras.layers.Dense(units=num_classes,activation="sigmoid"))
#Compiling ANN

ann.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=[tf.keras.metrics.AUC(), Precision(), Recall()])


# Fitting ANN
ann.fit(X_train, y_train, batch_size=30, epochs=10, validation_split=0.3)

# Metrics evaluation for ANN Model
metrics = evaluate_model(ann, X_test, y_test)

ann_result = {
    'Model': 'ANN',
    'AUC': metrics[0],
    'Precision': metrics[1],
    'Recall': metrics[2],
    'F1': metrics[3]
}

# Use pd.concat to add the new result to results DataFrame
results = pd.concat([results, pd.DataFrame([ann_result])], ignore_index=True)


def highlight_min(data, color='red'):
    # Create a DataFrame to store the styles
    attr = 'background-color: {}'.format(color)
    is_min = data == data.min().min()
    return pd.DataFrame(np.where(is_min, attr, ''), index=data.index, columns=data.columns)

def highlight_max(data, color='lightgreen'):
    # Create a DataFrame to store the styles
    attr = 'background-color: {}'.format(color)
    is_max = data == data.max().max()
    return pd.DataFrame(np.where(is_max, attr, ''), index=data.index, columns=data.columns)


results.style.apply(highlight_min, color='red', axis=None).apply(highlight_max, color='lightgreen', axis=None)

from matplotlib.colors import LinearSegmentedColormap

#cm = sns.light_palette("green", as_cmap=True)
cm = LinearSegmentedColormap.from_list(
    name='test', 
    #colors=['red','white','green','white','red']
    colors=['tomato','orange','white','lightgreen','green']
)

numeric_columns = results.select_dtypes(include=np.number).columns
t = results[numeric_columns].style.background_gradient(cmap=cm)

t

# Convert results dictionary to DataFrame
metrics_data = []

# Extract the data from the results DataFrame
for index, row in results.iterrows():
    metrics_data.append([row['Model'], row['AUC'], row['Precision'], row['Recall'], row['F1']])

metrics_df = pd.DataFrame(metrics_data, columns=['Model', 'AUC', 'Precision', 'Recall', 'F1'])

# Plot the results using bar plots for each metric
fig, axes = plt.subplots(2, 2, figsize=(20, 15))

# Plot AUC
sns.barplot(data=metrics_df, x='Model', y='AUC', ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title('AUC Comparison')
axes[0, 0].set_ylim(0, 1)

# Plot Precision
sns.barplot(data=metrics_df, x='Model', y='Precision', ax=axes[0, 1], palette='plasma')
axes[0, 1].set_title('Precision Comparison')
axes[0, 1].set_ylim(0, 1)

# Plot Recall
sns.barplot(data=metrics_df, x='Model', y='Recall', ax=axes[1, 0], palette='inferno')
axes[1, 0].set_title('Recall Comparison')
axes[1, 0].set_ylim(0, 1)

# Plot F1 Score
sns.barplot(data=metrics_df, x='Model', y='F1', ax=axes[1, 1], palette='magma')
axes[1, 1].set_title('F1 Score Comparison')
axes[1, 1].set_ylim(0, 1)

# Set common formatting for all plots
for ax in axes.flat:
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


import nbformat

def extract_code_from_ipynb(ipynb_file, output_file):
    with open(ipynb_file, 'r', encoding='utf-8') as file:
        notebook = nbformat.read(file, as_version=4)
        
    # Extract only code cells
    code_cells = [cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code']
    
    # Write each code cell into output file
    with open(output_file, 'w', encoding='utf-8') as file:
        for code in code_cells:
            file.write(code)
            file.write('\n\n')

# Example use case to extract code into a .py file
extract_code_from_ipynb('multilabel-classification-cnn-dnn-lstm.ipynb', 'dl.py')



