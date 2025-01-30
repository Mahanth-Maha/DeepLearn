(Note : NLP notes now have a new repo here → [NLP notes](https://github.com/Mahanth-Maha/NLP-DS207/tree/main))

# Simple Auto Grad (like pytorch api)

I have implemented a simple auto grad library that is similar to the pytorch api.

implemented from scratch, with smallest node being value class $\texttt{nnValue}$, which extends $\texttt{neuronValue}$, which then used to create $\texttt{Neuron}$ with weights and bias, which is then used to create $\texttt{Layer}$, which is then used to create $\texttt{MultiLayer}$ network. 

### class hierarchy

$$
\texttt{nnValue} \\
\downarrow \\
\texttt{Neuron} \\
\downarrow \\
\texttt{Layer}\\ 
\downarrow \\
\texttt{MultiLayer}
$$

The gradient is calculated using the chain rule, and the backpropagation is done using the same in the value level, so it is very similar to the pytorch api. 

The forward pass is done by calling the $\texttt{forward}$ method of the $\texttt{MultiLayer}$ network, and the backward pass is done by calling the $\texttt{backward}$ method of the $\texttt{MultiLayer}$ network.

### passes 

$$
\texttt{MultiLayer} \\
\texttt{forward pass} \downarrow  \uparrow \texttt{backward pass} \\
\texttt{Loss}
$$


on calling the $\texttt{backward}$ method, the gradients are calculated and stored in the $\texttt{grad}$ attribute of the $\texttt{nnValue}$ class. 

check [Simple Grad Perceptron notebook](./simpleGradPerceptron.ipynb) for complete flow and implementation.