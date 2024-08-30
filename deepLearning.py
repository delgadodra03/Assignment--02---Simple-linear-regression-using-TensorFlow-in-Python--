# 1) Import standard libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 2) Generate a dataset
def generate_data(true_coeffs = np.array([10,1]), noise_std = 2, num_samples = 100):
    
    #Seed for reproducibility
    np.random.seed(4500)
    
    #Generate synthetic data
    x = np.arange(0,100,1)
    y = true_coeffs[0] + true_coeffs[1]*x + np.random.normal(loc = 0.0, scale = noise_std, size = num_samples)
    
    #Return features and response
    return (x,y)

#Generate a synthetic dataset for simple linear regression
true_coeffs = np.array([10,1])
noise_std = 5
x, y = generate_data(true_coeffs=true_coeffs, noise_std=noise_std, num_samples=100)

# 3) Explore the dataset (Exploratory Data Analysis EDA)
#Visualise the generated synthetica dataset
plt.figure(figsize=(10,7))
plt.scatter(x, y, label='Synthetic dataset')
plt.xlabel(r"$x$", fontsize = 20)
plt.ylabel("$f_{\mathbf{w}}(x)$", fontsize = 20)
plt.title(rf"$f_{{\mathbf{{w}}}}(x) = {true_coeffs[0]} + {true_coeffs[1]} x + \epsilon$, where $\epsilon \sim \mathcal{{N}}(\mu=0, \sigma={noise_std})$", fontsize = 20)
plt.legend()
plt.show()

# 4) Split the original dataset
#Split the data into training and testing sets using train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, shuffle = True)

#Visualise the generated synthetica dataset
plt.figure(figsize = (10,7))
plt.scatter(x, y, label='Testing dataset')
plt.scatter(x_train, y_train, label='Training dataset', color=[1,0,0])
plt.xlabel(r"$x$", fontsize = 20)
plt.ylabel("$f_{\mathbf{w}}(x)$", fontsize = 20)
plt.title(rf"$f_{{\mathbf{{w}}}}(x) = {true_coeffs[0]} + {true_coeffs[1]} x + \epsilon$, where $\epsilon\sim \mathcal{{N}}(\mu=0, \sigma={noise_std})$", fontsize = 20)
plt.legend()
plt.show()

# 5) Design the neural network architecture
#Build the linear regression model using a multiple-input single neuron
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

#Compile the model
model.compile(optimizer='adam', loss='mse')

model.summary()

# 6) Train the neural network model
#Train the model
model.fit(x_train, y_train, epochs=500, batch_size=10, validation_split=0.10, verbose=1)

#Evaluate the model
loss = model.evaluate(x_test, y_test)
print(f'\nTest Loss: {loss}')

#Print the widgets of the trained model
weights = model.layers[0].get_weights()
print(f"Weights [w1]: {weights[0]}")
print(f"Biases [w0]: {weights[1]}")

# 7) Make predictions
#Make predictions
y_pred = model.predict(x_test)

#Make predictions
y_pred_train = model.predict(x_train)

# 8) Visualise the learnt model
#Plot the results
plt.figure(figsize=(12,6))

plt.scatter(x_train, y_train, label='Training data')
plt.plot(x_train, y_pred_train, label='Model', linewidth=3, color=[1,0,0])
plt.xlabel(r'$x$', fontsize = 20)
plt.ylabel("$\hat{f}_{\mathbf{w}}(x)$", fontsize = 20)
plt.legend();