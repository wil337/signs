X_train_orig <- load_dataset()$train_set_x_orig
Y_train_orig <- load_dataset()$train_set_y_orig
X_test_orig <- load_dataset()$test_set_x_orig
Y_test_orig <- load_dataset()$test_set_y_orig
classes <- load_dataset()$classes

# Example of a picture
library(imager)
index = 4
newimg <- aperm(train_dataset["train_set_x"][index,,,],c(2,3,1,4)) #convert dimensions into usable for imager
plot(cimg(newimg))
dim(X_train_orig)

# Flatten the training and test images
X_train_flatten = matrix(X_train_orig, nrow=64*64*3)
X_test_flatten = matrix(X_test_orig, nrow=64*64*3)
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

cat ("number of training examples = " , dim(X_train)[2])
cat ("number of test examples = " , dim(X_test)[2])
cat ("X_train shape: " ,dim(X_train))
cat ("Y_train shape: " , dim(Y_train))
cat ("X_test shape: " , dim(X_test))
cat ("Y_test shape: " , dim(Y_test))

create_placeholders <- function(n_x, n_y){
  
  #  Creates the placeholders for the tensorflow session.
  
  #Arguments:
  #n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
  #n_y -- scalar, number of classes (from 0 to 5, so -> 6)
  
  #Returns:
  #X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
  #Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
  
  #Tips:
  #- You will use None because it let's us be flexible on the number of examples you will for the placeholders.
  #In fact, the number of examples during test/train is different.
  
  X = tf$placeholder(tf$float32, shape=shape(n_x, NULL),name="X")
  Y = tf$placeholder(tf$float32, shape = shape(n_y, NULL), name ="Y")
  
  return(list(X=X, Y=Y))
}
create_placeholders(12288, 6)


initialize_parameters <- function(){

  #Initializes parameters to build a neural network with tensorflow. The shapes are:
  #W1 : [25, 12288]
  #b1 : [25, 1]
  #W2 : [12, 25]
  #b2 : [12, 1]
  #W3 : [6, 12]
  #b3 : [6, 1]
  
  #Returns:
  #parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
  
  
  tf$set_random_seed(1)                   # so that your "random" numbers match ours
  
  W1 = tf$get_variable("W1", shape(25L,12288L), initializer=tf$contrib$layers$xavier_initializer(seed=1L))
  b1 = tf$get_variable("b1", shape(25L,1L),initializer=tf$zeros_initializer())
  W2 = tf$get_variable("W2", shape(12L,25L), initializer=tf$contrib$layers$xavier_initializer(seed=1))
  b2 = tf$get_variable("b2", shape(12L,1L), initializer=tf$zeros_initializer())
  W3 = tf$get_variable("W3", shape(6,12), initializer=tf$contrib$layers$xavier_initializer(seed=1))
  b3 = tf$get_variable("b3", shape(6,1), initializer=tf$zeros_initializer())

  
  parameters = list(W1=W1, b1 = b1, W2=W2,b2= b2,W3=W3,b3=b3)
  
  return(parameters)
}
tf$reset_default_graph()
with(tf$Session() %as% sess,{
    parameters <- initialize_parameters()
    print(parameters$W1)
    print(parameters$b1)
    print(parameters$W2)
    print(parameters$b2)
})

forward_propagation <- function(X, parameters){
  
  #Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
  
  #Arguments:
  #  X -- input dataset placeholder, of shape (input size, number of examples)
  #parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
  #the shapes are given in initialize_parameters
  
  #Returns:
  #  Z3 -- the output of the last LINEAR unit
  
  
  # Retrieve the parameters from the dictionary "parameters" 
  W1 <- parameters$W1
  b1 <- parameters$b1
  W2 <- parameters$W2
  b2 <- parameters$b2
  W3 <- parameters$W3
  b3 <- parameters$b3
  
                                                         # Numpy Equivalents:
  Z1 <- tf$add(tf$matmul(W1,X),b1)                         # Z1 = np.dot(W1, X) + b1
  A1 <- tf$nn$relu(Z1)                                     # A1 = relu(Z1)
  Z2 <- tf$add(tf$matmul(W2,A1),b2)                        # Z2 = np.dot(W2, a1) + b2
  A2 <- tf$nn$relu(Z2)                                     # A2 = relu(Z2)
  Z3 <- tf$add(tf$matmul(W3,A2),b3)                        # Z3 = np.dot(W3,Z2) + b3

  
  return(Z3)
}
tf$reset_default_graph()

with(tf$Session() %as% sess,{
  X<-create_placeholders(12288, 6)$X
  Y<-create_placeholders(12288, 6)$Y
  parameters <- initialize_parameters()
  Z3 <- forward_propagation(X, parameters)
  print(Z3)
})

compute_cost <-function(Z3, Y){
  #Arguments:
  #  Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
  #Y -- "true" labels vector placeholder, same shape as Z3
  
  #Returns:
  #  cost - Tensor of the cost function
  
  
  # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
  logits <- tf$transpose(Z3)
  labels <- tf$transpose(Y)
  
  cost <- tf$reduce_mean(tf$nn$softmax_cross_entropy_with_logits(logits = logits, labels = labels))
  return(cost)  
}
tf$reset_default_graph()
with(tf$Session() %as% sess,{
  X <- create_placeholders(12288, 6)$X
  Y <- create_placeholders(12288, 6)$Y
  parameters <- initialize_parameters()
  Z3 <- forward_propagation(X, parameters)
  cost <- compute_cost(Z3, Y)
  print(cost)
})


model <- function(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500L, minibatch_size = 32L, print_cost = TRUE){

  #Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
  
  #Arguments:
  #X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
  #Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
  #X_test -- training set, of shape (input size = 12288, number of training examples = 120)
  #Y_test -- test set, of shape (output size = 6, number of test examples = 120)
  #learning_rate -- learning rate of the optimization
  #num_epochs -- number of epochs of the optimization loop
  #minibatch_size -- size of a minibatch
  #print_cost -- True to print the cost every 100 epochs
  
  #Returns:
  #parameters -- parameters learnt by the model. They can then be used to predict.
  
  tf$reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
  tf$set_random_seed(1)                             # to keep consistent results
  seed = 3L                                          # to keep consistent results
  n_x <- dim(X_train)[1]                            # (n_x: input size,
  m <- dim(X_train)[2]                              #m : number of examples in the train set)
  n_y = dim(Y_train)[1]                            # n_y : output size
  costs = matrix()                                        # To keep track of the cost
  
  # Create Placeholders of shape (n_x, n_y)
  X <- create_placeholders(n_x,n_y)$X
  Y <- create_placeholders(n_x,n_y)$Y
  
  # Initialize parameters
  parameters <- initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
  Z3 <- forward_propagation(X, parameters)

  # Cost function: Add cost function to tensorflow graph
  cost <- compute_cost(Z3, Y)

  # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
  optimizer <-  tf$train$GradientDescentOptimizer(learning_rate = learning_rate)$minimize(cost)
  
  # Initialize all the variables
  init <- tf$global_variables_initializer()
  
  # Start the session to compute the tensorflow graph
  with(tf$Session() %as% sess,{
    #sess <- tf$Session()
    # Run the initialization
    sess$run(init)
    # Do the training loop
    for (epoch in 1:num_epochs){
      epoch_cost <- 0                       # Defines a cost related to an epoch
      num_minibatches <- floor(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
      seed <- seed + 1L
      minibatches <- random_mini_batches(X_train, Y_train, mini_batch_size, seed)
      
      for (i in 1:length(minibatches$mini_batch_X)){
        # Select a minibatch
        minibatch_X <- minibatches$mini_batch_X[[i]]
        minibatch_Y <- minibatches$mini_batch_Y[[i]]
        
        # IMPORTANT: The line that runs the graph on a minibatch.
        # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
        
        minibatch_optimizer = sess$run(optimizer, feed_dict=dict(X=minibatch_X, Y=minibatch_Y))
        minibatch_cost = sess$run(cost, feed_dict=dict(X=minibatch_X, Y=minibatch_Y))
        
        epoch_cost = epoch_cost + minibatch_cost / num_minibatches
        
      }
      
      # Print the cost every epoch
      if (print_cost == TRUE & epoch %% 100 == 0)
        cat ("Cost after epoch" , epoch,":", epoch_cost, "\n")
      if (print_cost == TRUE & epoch %% 5 == 0){
        if(epoch ==1) costs <- epoch_cost
        else costs<-cbind(costs,epoch_cost)
      }
    }    
        
    # plot the cost
    plot(costs,ylab="cost", xlab="iterations (per tens)",main=cat("learning rate = ",learning_rate))
    #plt.plot(np.squeeze(costs))
    #plt.ylabel('cost')
    #plt.xlabel('iterations (per tens)')
    #plt.title(Learning rate =" + str(learning_rate))
    #plt.show()
    
    # lets save the parameters in a variable
    parameters <- sess$run(parameters)
    print ("Parameters have been trained!")
    
    # Calculate the correct predictions
    correct_prediction <- tf$equal(tf$argmax(Z3), tf$argmax(Y))
    
    # Calculate accuracy on the test set
    accuracy <- tf$reduce_mean(tf$cast(correct_prediction, "float"))
    
    cat ("Train Accuracy:", sess$run(accuracy, dict(X=X_train, Y=Y_train)), "\n")
    cat ("Test Accuracy:", sess$run(accuracy, dict(X=X_test, Y=Y_test)),"\n")
      
    
  })#sess$close()
  return(parameters)  
}
parameters <- model(X_train, Y_train, X_test, Y_test,learning_rate = 0.0001, num_epochs = 1500L)
