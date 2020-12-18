import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Load the TensorBoard notebook extension

def plot_signal(signal):
    """
    this function will take the signal dictionary and plot the signals
    """
    fig , axes = plt.subplots(nrows=5 , ncols=2 , sharex =False ,sharey=True,
                             figsize=(40,20))
    fig.suptitle('Time series',size=15)
    i=0
    for x in range(5):
        for y in range(2):
            axes[x,y].set_title(list(signal.keys())[i])
            axes[x,y].plot(list(signal.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i +=1
    plt.show()

def dis_feature(mfccs, cmap=None):
    """
    this function will take the mfcc/mel_spectrogram dictionary and plot the signals
    """
    fig ,axes= plt.subplots(nrows=5 , ncols=2 , sharex=False, sharey=True , figsize=(40,20))
    fig.suptitle('mel')
    i=0
    for x in range(5):
        for y in range(2):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i], cmap=cmap,interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i+=1
    plt.show()

def visulization(sample_df):
    signals = {}
    mel_spectrograms = {}
    mfccs = {}

    for row in tqdm(sample_df.iterrows()):  # every row will be like [[index], [filename , target , category]]
        signal , rate = librosa.load(DATA_PATH+ row[1][0])
        signals[row[1][2]] = signal    # row[1][2] will be the category of that signal. eg. signal["dog"] = signal of dog sound
        
        mel_spec = librosa.feature.melspectrogram(y=signal , sr=rate ,  n_fft=2048, hop_length=512)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  #visualizing mel_spectrogram directly gives black image. So, coverting from power_to_db is required
        mel_spectrograms[row[1][2]] = mel_spec
        
        mfcc = librosa.feature.mfcc(signal , rate , n_mfcc=13, dct_type=3)
        mfccs[row[1][2]] = mfcc
    plot_signal(signals)
    dis_feature(mel_spectrograms)
    dis_feature(mfccs, cmap='hot')

def sample_data(df, sample_len, sample_num, num_classes):
    X , y = [] , []
    for data in tqdm(df.iterrows()):
        sig , sr = librosa.load(DATA_PATH+data[1][0])
        for i in range(sample_num):
                n = np.random.randint(0, len(sig)-(sr*sample_len))
                sig_ = sig[n : int(n+(sr*sample_len))]
                # here use mfcc feature
                mfcc_ = librosa.feature.mfcc(sig_ , sr=sr, n_mfcc=13)
                # could also use mel slectrogram feature
                #mel_spec = librosa.feature.melspectrogram(y=sig_ , sr=sr)
                #mel_spec_ = librosa.power_to_db(mel_spec, ref=np.max)  #visualizing mel_spectrogram directly gives black image. So, coverting from power_to_db is required
                #X.append(mel_spec_)
                X.append(mfcc_)
                y.append(data[1][1])

    # convert list to numpy array
    X = np.array(X) 
    y = np.array(y)

    #one-hot encoding the target
    #y = tf.keras.utils.to_categorical(y , num_classes=num_classes)

    # our tensorflow model takes input as (no_of_sample , height , width , channel).
    # here X has dimension (no_of_sample , height , width).
    # So, the below code will reshape it to (no_of_sample , height , width , 1).
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    x_train , x_val , y_train , y_val = train_test_split(X , y ,test_size=0.2, random_state=2020)
    train_x_name = 'binary2smfcc_train_x.npy'
    train_y_name = 'binary2smfcc_train_y.npy'
    test_x_name = 'binary2smfcc_test_x.npy'
    test_y_name = 'binary2smfcc_test_y.npy'
    np.save(train_x_name, x_train)
    np.save(train_y_name, y_train)
    np.save(test_x_name,x_val)
    np.save(test_y_name, y_val)

def load_train_test_data():
    train_x = np.load('binary2smfcc_train_x.npy')
    train_y = np.load('binary2smfcc_train_y.npy')
    test_x = np.load('binary2smfcc_test_x.npy')
    test_y = np.load('binary2smfcc_test_y.npy')
    return train_x, train_y, test_x, test_y

def train_model(train_x, train_y, test_x, test_y, model):
    LOGDIR = "binary_logs"
    CPKT = "binary_cpkt/cp-{epoch:04d}.ckpt"
    #this callback is used to prevent overfitting.
    callback_1 = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto',
        baseline=None, restore_best_weights=False
    )

    #this checkpoint saves the best weights of model at every epoch
    callback_2 = tf.keras.callbacks.ModelCheckpoint(
        CPKT, monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch', options=None
    )

    #this is for tensorboard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGDIR)

    # Save the weights using the `checkpoint_path` format
    model.save_weights(CPKT.format(epoch=0))


    history = model.fit(train_x,train_y,
        validation_data=(test_x,test_y),
        epochs=100,
        callbacks = [callback_1 , callback_2 , tensorboard_callback])

def build_model(train_x, train_y, test_x, test_y, num_classes):
    #train_y = tf.keras.utils.to_categorical(train_y, num_classes=num_classes)
    #test_y = tf.keras.utils.to_categorical(test_y, num_classes=num_classes)
    #INPUTSHAPE = (13,87,1)
    INPUTSHAPE = train_x[0].shape
    model = create_model(INPUTSHAPE, num_classes)
    print(model.summary())
    train_model(train_x, train_y,test_x, test_y, model)

def predict_model(test_x, test_y, num_classes):
    INPUTSHAPE = test_x[0].shape
    model = create_model(INPUTSHAPE, num_classes)
    # Restore the weights
    latest = tf.train.latest_checkpoint('binary_cpkt')
    model.load_weights(latest)
    loss, acc = model.evaluate(test_x, test_y, verbose=2)
    predicted_class = model.predict_classes(test_x)
    predicted_class = predicted_class.flatten()
    temp = sum(test_y== predicted_class)
    print(temp/len(test_y))
    # using jupyter notebook for sns
    #cm = confusion_matrix(test_y, predicted_class)
    #sns.heatmap(cm, annot=True)

def create_model(INPUTSHAPE, num_classes):
    model =  tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16 , (3,3),activation = 'relu',padding='valid', input_shape = INPUTSHAPE),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu',padding='valid'),

        tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='valid'),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='valid'),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='valid'),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='valid'),
        tf.keras.layers.GlobalAveragePooling2D(),
        
        tf.keras.layers.Dense(32 , activation = 'relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation = tf.nn.sigmoid)
    ])
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    return model

    



if __name__ == '__main__':
    DATA_PATH = '../en_sound50/audio/audio/'
    CSV_FILE_PATH = '../en_sound50/esc50.csv'
    df = pd.read_csv(CSV_FILE_PATH)
    chose_label = ['dog','chirping_birds','thunderstorm','sheep','water_drops',
    'wind','footsteps','frog','cow','rain','insects','breathing','cat']
    animal = ['dog', 'chirping_birds','sheep','footsteps','frog','cow','insects','cat']
    non_animal = ['thunderstorm','water_drops','wind','rain','breathing']
    class_dict = {i:x for x,i in enumerate(chose_label)}
    # set animal to 1 and non-animal to 0
    for tmp in animal:
        class_dict[tmp] = 1
    for tmp in non_animal:
        class_dict[tmp] = 0
    num_classes = 2

    # chose label dataframe
    df_chosen = df[df['category'].isin(chose_label)]
    df_chosen = df_chosen.drop(['fold','esc10','src_file','take'], axis=1)
    df_chosen['target'] = df_chosen['category'].map(class_dict)
    
    '''
    # if do not need to visulize the data, just comment these
    # chose 1 per label for visulization
    sample_df = df_chosen.drop_duplicates(subset=['target'])
    
    visulization(sample_df)
    '''
    '''
    # random sample data from sound
    # if already built the dataset, comment these
    sample_len = 2 # sounds for each training x
    sample_num = 3 # random sample # samples in total 5 seconds
    # usually sample_len * sample_num = 5
    sample_data(df_chosen, sample_len, sample_num, num_classes)
    '''
    
    train_x, train_y, test_x, test_y = load_train_test_data()
    #build_model(train_x, train_y, test_x, test_y, num_classes)


    # prediction pipeline
    predict_model(test_x, test_y, num_classes)






        

