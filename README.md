# SafeAD


The sole purpose of this repository is to provide solutions to the two tasks given as part of selection criteria. 

Can you think of another similar task where the positional encoding would not be necessary?


Positional encoding is not required in case of image classification. In image classification we do not care about the position of the object in image but only about presence of an object.


Summarize the idea of attention in a few sentences. What are Q, K and V?

Attention is a process of focusing on few things ignoring many others. It is a mode of mechanism domain of computer vision where we only concentrate on few objects in the image relevant to task and ignoring other objects- 

Q is a query, K is a key and V is value. The attention mechanism compares the query with the keys and get weights of values. Then the weightd are reweighted by the network and considers the total of reweighed weights. Each weight represents the degree of correspondence between the query and the key.
