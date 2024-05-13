# Data Science (HW3) - 109006234

## Installing the DGL Package
No special implementation. Just run the cell.

## Replacing the Traditional Command Line Arguments
This cell's purpose is to the place the command line arguments if the program were to be run in the the device's terminal. Due to the computation limitation of my device, for this homework I will be using Google Colab and the program is in a Jupyter Notebook format. Just run the cell as usual and if we were to change the parameters, we can edit the `args` variable.

## Load Data from the Provided Dataset
As it is based on the provided sample code, no changes made for this part. Just run the cell as usual.

## GAT Model
The Graph Attention Network (GAT) model comprises multiple layers, including input, hidden, and output layers, each performing graph attention operations. This design allows the model to capture intricate relationships and hierarchies present in the graph. 

During the forward pass, the GAT model processes input features along with the graph structure through a sequence of GATConv layers, applying attention mechanisms to aggregate information from neighboring nodes. The incorporation of an activation function introduces non-linearity to the model, enhancing its ability to capture complex relationships in the data. Moreover, dropout regularization is applied to prevent overfitting by randomly zeroing out activations during training, ensuring robustness and generalization to unseen data. Ultimately, the output predictions produced by the final layer of the GAT model encapsulate the model's understanding of the graph data, providing valuable insights and enabling informed decision-making based on the learned patterns and relationships.

For this part, we also experimented with several activation function to try to improve the performance, such as ReLU, LeakyReLu, etc. However, only ELU gives us the best performance.

## Evaluating and Training the Model
As it is based on the provided sample code, no changes made for this part. Just run the cell as usual.

## Driver Function
In this cell, we modified `in_size` to represent the input features of each node. It is the number of columns in the `features` tensor. Then we set the hidden layer to be set to 32. Next, the size of the output layer is set to be determined based on `num_classes`. Lastly, the `num_heads` represents the separate attention mechanism to capture different aspect of the node's relationship with each other, 'num_layers' represent how many GAT layers are there and `dropout` is introduced to avoid overfitting. 

Other than that, as it is based on the provided sample code, no changes made for this part. Just run the cell as usual.

## Efforts to Improve Performance
### Hyperparameter Tuning
`model = GAT(in_size, 32, out_size, num_heads=4, num_layers=2, dropout=0.1)`
These tuning was found to be the best performing one after doing several experiments. With random search, we experimented on several options and combinations of the tuning and this set yields the best result.
### Other Attempts to Improve Performance
We experimented on models other than GAT, which includes GIN, GraphSAGE, GC-LSTM, however none resulted to an improved result. So we stick to using the GAT model. We also tried to implement learning rate scheduler, L1/L2 regularization, graph augmentation, yet only hyperparameter tuning gives us the best result.
