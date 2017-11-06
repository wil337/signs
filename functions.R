library(reticulate)
library(tensorflow)
linear_function<-function(){
    np <- import("numpy")
  
#Implements a linear function: 
#  Initializes W to be a random tensor of shape (4,3)
#  Initializes X to be a random tensor of shape (3,1)
#  Initializes b to be a random tensor of shape (4,1)
#  Returns: 
#  result -- runs the session for Y = WX + b 
    np$random$seed(1L)
    X <- tf$constant(np$random$randn(3L,1L), name="X")
    W <- tf$constant(np$random$randn(4L,3L), name="W")
    b <- tf$constant(np$random$randn(4L,1L), name="b")
    Y <- tf$constant(np$random$randn(4L,3L),  name="Y")

    # Create the session using tf.Session() and run it with sess.run(...) on the variable you want to calculate
    
    sess <- tf$Session()
    result <- sess$run(tf$add(tf$matmul(W, X),b))

    
    # close the session 
    sess$close()

    return(result)  
}

print(linear_function())


sigmoid<-function(z){
  
  #Computes the sigmoid of z
  
  #Arguments:
  #z -- input value, scalar or vector
  
  #Returns: 
  #results -- the sigmoid of z
  
  # Create a placeholder for x. Name it 'x'.
  x <- tf$placeholder(tf$float32,name='x')
  
  # compute sigmoid(x)
  sigmoid <- tf$sigmoid(x)
  
  # Create a session, and run it. Please use the method 2 explained above. 
  # You should use a feed_dict to pass z's value to x. 
  with(tf$Session() %as% sess, {
    # Run session and call the output "result"
    result <- sess$run(sigmoid,feed_dict=dict(x=z))
  })
  return(result)
}
print(sigmoid(0))
sigmoid(12)

cost <- function(logits,labels){
  #Computes the cost using the sigmoid cross entropy
  
  #Arguments:
  #  logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
  #labels -- vector of labels y (1 or 0) 
  
  #Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels" 
  #in the TensorFlow documentation. So logits will feed into z, and labels into y. 
  
  #Returns:
  #cost -- runs the session of the cost (formula (2))
  # Create the placeholders for "logits" (z) and "labels" (y) (approx. 2 lines)
  z <- tf$placeholder(tf$float32,name="z")
  y = tf$placeholder(tf$float32,name="y")
  
  # Use the loss function (approx. 1 line)
  cost <- tf$nn$sigmoid_cross_entropy_with_logits(logits=z,labels=y)
  
  # Create a session (approx. 1 line). See method 1 above.
  sess <- tf$Session()
  
  # Run the session (approx. 1 line).
  cost <- sess$run(cost, feed_dict=dict(z=logits, y=labels))
  
  # Close the session (approx. 1 line). See method 1 above.
  sess$close()
  
  return(cost)  
}
logits <- sigmoid(matrix(c(0.2,0.4,0.7,0.9)))
cost <- cost(logits, matrix(c(0,0,1,1)))
cat("cost = ", cost)


one_hot_matrix <- function(labels, C){
  #Creates a matrix where the i-th row corresponds to the ith class number and the jth column
  #corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
  #will be 1. 
  
  #Arguments:
  #  labels -- vector containing the labels 
  #C -- number of classes, the depth of the one hot dimension
  
  #Returns: 
  #  one_hot -- one hot matrix
  # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
  C <- tf$constant(C,name='C')
  
  # Use tf.one_hot, be careful with the axis (approx. 1 line)
  one_hot_matrix <- tf$one_hot(indices=labels,depth=C,axis=0L)
  
  # Create the session (approx. 1 line)
  sess <- tf$Session()
  
  # Run the session (approx. 1 line)
  one_hot <- sess$run(one_hot_matrix)
  
  # Close the session (approx. 1 line). See method 1 above.
  sess$close()
  return(one_hot)
  
  
}
labels <-array(c(1L,2L,3L,0L,2L,1L))
one_hot <- one_hot_matrix(labels, C = 4L)
one_hot

ones <-function(shape){
  #Creates an array of ones of dimension shape
  
  #Arguments:
  #  shape -- shape of the array you want to create
  
  #Returns: 
  #  ones -- array containing only ones
  
  # Create "ones" tensor using tf.ones(...). (approx. 1 line)
  ones <- tf$ones(shape)
  
  # Create the session (approx. 1 line)
  sess <- tf$Session()
  
  # Run the session to compute 'ones' (approx. 1 line)
  ones <- sess$run(ones)
  
  # Close the session (approx. 1 line). See method 1 above.
  sess$close()
  
  return(ones)  
}
ones(3)

#test import
h5py<-import("h5py")
np<-import("numpy")
test_dataset_py = h5py$File('test_signs.h5', "r")
test_set_x_orig_py = np$array(test_dataset["test_set_x"][]) # your test set features
test_set_y_orig_py = np$array(test_dataset["test_set_y"][]) # your test set labels

head(test_set_x_orig,500)-head(test_set_x_orig_py,500)
test_set_x_orig-test_set_x_orig_py

load_dataset <- function(){
  #h5py<-import("h5py")
  np<-import("numpy", convert=TRUE)
  #test_dataset = h5py$File('test_signs.h5', "r")
  test_dataset <- h5file('test_signs.h5', "r")
  test_set_x_orig = np$array(test_dataset["test_set_x"][]) # your test set features
  test_set_y_orig = np$array(test_dataset["test_set_y"][]) # your test set labels
  
  #train_dataset = h5py$File('train_signs.h5', "r")
  train_dataset <- h5file('train_signs.h5', "r")
  train_set_x_orig = np$array(train_dataset["train_set_x"][]) # your train set features
  train_set_y_orig = np$array(train_dataset["train_set_y"][]) # your train set labels
  
  #train_dataset <- h5file('train_signs.h5', "r")
  #train_set_x_orig <- array(train_dataset["train_set_x"][],c(1080,64,64,3)) # your train set features
  #train_set_y_orig = array(train_dataset["train_set_y"][]) # your train set labels
  
  #test_dataset <- h5file('test_signs.h5', "r")
  #test_set_x_orig = array(test_dataset["test_set_x"][],c(120,64,64,3)) # your test set features
  #test_set_y_orig = array(test_dataset["test_set_y"][]) # your test set labels
  
  #classes = array(test_dataset["list_classes"][]) # the list of classes
  classes = np$array(test_dataset["list_classes"][]) # the list of classes
  
  #train_set_y_orig = matrix(train_set_y_orig,ncol=1)
  #test_set_y_orig = matrix(test_set_y_orig,ncol=1)
  
  return(list(train_set_x_orig=train_set_x_orig, train_set_y_orig=train_set_y_orig, test_set_x_orig=test_set_x_orig, test_set_y_orig=test_set_y_orig, classes=classes))
  
}
load_dataset()$classes

random_mini_batches <- function(X, Y, mini_batch_size = 64L, seed = 0L){
  
  #Creates a list of random minibatches from (X, Y)
  
  #Arguments:
  #  X -- input data, of shape (input size, number of examples)
  #Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
  #mini_batch_size - size of the mini-batches, integer
  #seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
  
  #Returns:
  #mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
  
  
  m <- ncol(X)                  # number of training examples
  
  np$random$seed(seed)
  # Step 1: Shuffle (X, Y)
  permutation <- np$random$permutation(m)+1
  shuffled_X <- X[, permutation, drop=FALSE]
  shuffled_Y <- Y[, permutation, drop=FALSE]
  
  # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
  num_complete_minibatches <- floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
  mini_batch_X <- list(shuffled_X[, 1 : mini_batch_size, drop = FALSE])
  mini_batch_Y <- list(shuffled_Y[, 1 : mini_batch_size, drop = FALSE])
  k<-1
  while( k <= num_complete_minibatches-1){
    mini_batch_X[[k+1]] <- shuffled_X[, (k * mini_batch_size + 1) : (k * mini_batch_size + mini_batch_size), drop = FALSE]
    mini_batch_Y[[k+1]] <- shuffled_Y[, (k * mini_batch_size + 1) : (k * mini_batch_size + mini_batch_size), drop = FALSE]
    k=k+1
  }
  mini_batches <- list(mini_batch_X=mini_batch_X, mini_batch_Y=mini_batch_Y)
  
  # Handling the end case (last mini-batch < mini_batch_size)
  if (m %% mini_batch_size != 0) {
    mini_batch_X[[k+1]] <- shuffled_X[, (num_complete_minibatches * mini_batch_size+1) : m, drop = FALSE]
    mini_batch_Y[[k+1]] <- shuffled_Y[, (num_complete_minibatches * mini_batch_size+1) : m, drop = FALSE]
  }
  mini_batches <- list(mini_batch_X=mini_batch_X, mini_batch_Y=mini_batch_Y)
  return(mini_batches)
  
}
r<-random_mini_batches(X_train, Y_train, 1080L, seed = 0L)
str(r)

convert_to_one_hot <- function(Y, C){
  Y <- diag(C)[,Y+1]
  return(Y)
}
convert_to_one_hot(Y,5)

predicttf <-function(X, parameters){
  W1 = tf$convert_to_tensor(parameters["W1"])
  b1 = tf$convert_to_tensor(parameters["b1"])
  W2 = tf$convert_to_tensor(parameters["W2"])
  b2 = tf$convert_to_tensor(parameters["b2"])
  W3 = tf$convert_to_tensor(parameters["W3"])
  b3 = tf$convert_to_tensor(parameters["b3"])
  params <- list(w1=W1,b1=b1,W2=W2, b2=b2, W3=W3, b3=b3)
  
  x <- tf$placeholder("float", shape=shape(12288L, 1L))
  z3 = forward_propagation_for_predict(x, params)
  p = tf$argmax(z3)
  sess = tf$Session()
  prediction = sess$run(p, feed_dict = dict(x=X))
  
  return(prediction)  
}


forward_propagation_for_predict <- function(X, parameters){
  #Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
  
  #Arguments:
  #  X -- input dataset placeholder, of shape (input size, number of examples)
  #parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
  #the shapes are given in initialize_parameters
  
  #Returns:
  #  Z3 -- the output of the last LINEAR unit
  
  # Retrieve the parameters from the dictionary "parameters" 
  W1 = parameters['W1']
  b1 = parameters['b1']
  W2 = parameters['W2']
  b2 = parameters['b2']
  W3 = parameters['W3']
  b3 = parameters['b3'] 
  # Numpy Equivalents:
  Z1 = tf$add(tf$matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
  A1 = tf$nn$relu(Z1)                                    # A1 = relu(Z1)
  Z2 = tf$add(tf$matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
  A2 = tf$nn$relu(Z2)                                    # A2 = relu(Z2)
  Z3 = tf$add(tf$matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
  
  return(Z3)
  
}
