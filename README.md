import numpy as np
import pickle

# Assume loaded_model is your pre-trained model
# Replace this with your actual model loading code
loaded_model = None  # Replace with your model loading code

# Save the model to a file named 'model.pkl'
with open('model.pkl', 'wb') as model_file:
    pickle.dump(loaded_model, model_file)

# Now, let's use the saved model to make predictions on new input data
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Convert input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Load the saved model from 'model.pkl'
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Make predictions using the loaded model
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')

