# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

It consists of an input layer with 1 neuron, two hidden layers with 8 neurons in the first layer and 10 neurons in the second layer, and an output layer with 1 neuron. Each neuron in one layer is connected to all neurons in the next layer, allowing the model to learn complex patterns. The hidden layers use activation functions such as ReLU to introduce non-linearity, enabling the network to capture intricate relationships within the data. During training, the model adjusts its weights and biases using optimization techniques like RMSprop or Adam, minimizing a loss function such as Mean Squared Error for regression.The forward propagation process involves computing weighted sums, applying activation functions, and passing the transformed data through layer.

## Neural Network Model

<img width="884" height="490" alt="image" src="https://github.com/user-attachments/assets/fecba2c7-07c1-41e9-b420-2ed3513b0912" />

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: gokul prakash M
### Register Number: 212223240041
```python
gokul_brain = NeuralNet()
criterion=nn.MSELoss()
optimizer=torch.optim.RMSprop(gokul_brain.parameters(),lr=0.001)

def train_model(gokul_brain,X_train,y_train,criterion,optimizer,epochs=2000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    loss = criterion(gokul_brain(X_train),y_train)
    loss.backward()
    optimizer.step()

    gokul_brain.history['loss'].append(loss.item())
    if epoch % 200 == 0:
      print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

train_model(gokul_brain,X_train_tensor,Y_train_tensor,criterion,optimizer)


```
## Dataset Information
<img width="154" height="626" alt="image" src="https://github.com/user-attachments/assets/4340cd76-d98a-408d-86d6-f7e227eb4c7c" />

## OUTPUT

### Training Loss Vs Iteration Plot
<img width="290" height="211" alt="image" src="https://github.com/user-attachments/assets/45045cd8-06ce-4531-b2a9-88f78c04e5e3" />
<img width="590" height="458" alt="image" src="https://github.com/user-attachments/assets/e83d95bf-887f-47ac-84c1-66f778829338" />



## RESULT
To develop a neural network regression model for the given dataset is excuted sucessfully.
