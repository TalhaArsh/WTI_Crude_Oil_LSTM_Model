
This project focuses on predicting future WTI Crude Oil prices using a Long Short-Term Memory (LSTM) model, a type of Recurrent Neural Network (RNN) particularly well-suited for time series forecasting. The code is written in Python, utilizing popular libraries such as pandas, NumPy, scikit-learn, and Keras (with TensorFlow as the backend). The dataset used consists of historical WTI crude oil prices.

The dataset (WTI_CRUDE_YEAR.csv) contains historical WTI Crude Oil prices with the following columns:

time: Date of the recorded price (in YYYY-MM-DD format)
price: The corresponding price of WTI crude oil in USD.
The dataset is loaded into a pandas DataFrame, sorted by date, and then normalized for input into the LSTM model.

Preprocessing
Data Normalization:

The prices are normalized using MinMaxScaler to scale the values between 0 and 1. This is important for improving the performance of the neural network.
Create Dataset:

A helper function create_dataset is used to create the input-output pairs. The model is trained to predict the price at a given time based on the previous 60 days (look-back period).
The dataset is split into training (80%) and testing (20%) sets.
Reshaping:

The input data is reshaped to a 3D format ([samples, time steps, features]) to be suitable for the LSTM model.
LSTM Model
The LSTM model is built using Keras:

Model Architecture:

The model consists of three LSTM layers, each with 50 units, followed by Dropout layers to prevent overfitting.
The output layer is a Dense layer with a single unit, corresponding to the predicted price.
Training:

The model is compiled with the Adam optimizer and mean squared error loss function.
Early stopping is applied to avoid overfitting, with training stopping if the validation loss does not improve for 10 epochs.
The model is trained using the training data, with a batch size of 16 and for up to 100 epochs.
Prediction and Evaluation:

After training, the model is used to predict prices on the test data.
The predicted prices are then inverse transformed to their original scale for comparison with actual prices.
Performance metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) are calculated to evaluate the model's accuracy.
Visualization:

A plot is generated to visualize the actual vs. predicted prices, highlighting the model's performance.
Model Saving:

The trained model is saved as lstm_model.keras for future use.
