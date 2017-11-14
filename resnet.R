library(keras)

identity_block <- function(X, f, filters, stage, block){
#Implementation of the identity block as defined in Figure 3
  
#  Arguments:
#  X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
#  f -- integer, specifying the shape of the middle CONV's window for the main path
#  filters -- python list of integers, defining the number of filters in the CONV layers of the main path
#  stage -- integer, used to name the layers, depending on their position in the network
#  block -- string/character, used to name the layers, depending on their position in the network
  
#  Returns:
#  X -- output of the identity block, tensor of shape (n_H, n_W, n_C)

  # defining name basis
  conv_name_base <- paste0('res', as.character(stage), block, '_branch')
  bn_name_base <- paste0('bn', as.character(stage), block, '_branch')
  
  # Retrieve Filters
  F1 <- filters[1]
  F2 <- filters[2]
  F3 <- filters[3]
  
  # Save the input value. You'll need this later to add back to the main path. 
  X_shortcut <- X
  
  # First component of main path
  X <- X %>% 
    layer_conv_2d(filters = F1, kernel_size = c(1L, 1L), strides = c(1L,1L), 
                  padding = 'valid', name = paste0(conv_name_base, '2a'), 
                  kernel_initializer = "glorot_uniform") %>% 
    layer_batch_normalization(axis = 3L, name = paste0(bn_name_base, '2a')) %>%
    layer_activation('relu') %>% 
  # Second component of main path (≈3 lines)
    layer_conv_2d(filters = F2, kernel_size = c(f, f), strides = c(1L,1L), 
                  padding = 'same', name = paste0(conv_name_base, '2b'), 
                  kernel_initializer = "glorot_uniform") %>% 
    layer_batch_normalization(axis = 3L, name = paste0(bn_name_base, '2b')) %>% 
    layer_activation('relu') %>% 
  # Third component of main path (≈2 lines)
    layer_conv_2d(filters = F3, kernel_size = c(1L, 1L), strides = c(1L,1L), 
                  padding = 'valid', name = paste0(conv_name_base, '2c'), 
                  kernel_initializer = "glorot_uniform") %>% 
    layer_batch_normalization(axis = 3L, name = paste0(bn_name_base, '2c'))
  # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
  X <- layer_add(list(X, X_shortcut)) %>% 
    layer_activation('relu')
  
  return(X)
  
}

convolutional_block <- function(X, f, filters, stage, block, s = 2L){
# Implementation of the convolutional block as defined in Figure 4
#  Arguments:
#    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
#   f -- integer, specifying the shape of the middle CONV's window for the main path
#    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
#    stage -- integer, used to name the layers, depending on their position in the network
#    block -- string/character, used to name the layers, depending on their position in the network
#    s -- Integer, specifying the stride to be used
#    Returns:
#    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
  
  # defining name basis
  conv_name_base <- paste0('res', as.character(stage), block, '_branch')
  bn_name_base <- paste0('bn', as.character(stage), block, '_branch')
  
  # Retrieve Filters
  F1 <- filters[1]
  F2 <- filters[2]
  F3 <- filters[3]
  
  # Save the input value
  X_shortcut <- X
  
  ##### MAIN PATH #####
  # First component of main path 
  X <- X %>% 
    layer_conv_2d(F1, c(1L, 1L), strides = c(s,s), 
                     padding = 'valid', name = paste0(conv_name_base, '2a'), 
                     kernel_initializer = "glorot_uniform") %>% 
    layer_batch_normalization(axis = 3L, name = paste0(bn_name_base, '2a')) %>% 
    layer_activation('relu') %>% 
  # Second component of main path (≈3 lines)
    layer_conv_2d(F2, c(f, f), strides = c(1L,1L), 
                  padding = 'same', name = paste0(conv_name_base, '2b'), 
                  kernel_initializer = "glorot_uniform") %>% 
    layer_batch_normalization(axis = 3L, name = paste0(bn_name_base, '2b')) %>% 
    layer_activation('relu') %>% 
  # Third component of main path (≈2 lines)
    layer_conv_2d(F3, c(1L, 1L), strides = c(1L,1L), padding = 'valid', 
                  name = paste0(conv_name_base, '2c'), 
                  kernel_initializer = "glorot_uniform") %>%
    layer_batch_normalization(axis = 3L, name = paste0(bn_name_base, '2c'))
  ##### SHORTCUT PATH #### (≈2 lines)
  X_shortcut <- X_shortcut %>% 
    layer_conv_2d(F3, c(1L, 1L), strides = c(s,s), padding = 'valid', 
                  name = paste0(conv_name_base, '1'), 
                  kernel_initializer = "glorot_uniform") %>% 
    layer_batch_normalization(axis = 3L, name = paste0(bn_name_base, '1'))
  
  # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
  X <- layer_add(list(X_shortcut, X)) %>%
    layer_activation('relu')
  
  return(X)
}

resnet50 <- function(input_shape = c(64L, 64L, 3L), classes = 6L){
#  Implementation of the popular ResNet50 the following architecture:
#  CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
#  -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
#  Arguments:
#  input_shape -- shape of the images of the dataset
#  classes -- integer, number of classes
#  Returns:
#  model -- a Model() instance in Keras
  
  
  # Define the input as a tensor with shape input_shape
  X_input <- layer_input(input_shape)
  
  
  # Zero-Padding
  X <- X_input %>% 
    layer_zero_padding_2d(c(3L, 3L)) %>% 
  # Stage 1
    layer_conv_2d(filters=64L, 
                    kernel_size = c(7L, 7L), 
                    strides = c(2L, 2L), 
                    name = 'conv1', 
                    kernel_initializer = "glorot_uniform") %>% 
    layer_batch_normalization(axis = 3L, name = 'bn_conv1') %>% 
    layer_activation('relu') %>% 
    layer_max_pooling_2d(c(3L, 3L), strides=c(2L, 2L)) %>% 
  # Stage 2
    convolutional_block(3L, filters = c(64L, 64L, 256L), stage = 2L, 
                        block='a', s = 1L) %>% 
    identity_block(3L, c(64L, 64L, 256L), stage=2L, block='b') %>% 
    identity_block(3L, c(64L, 64L, 256L), stage=2L, block='c') %>% 
  
  ### START CODE HERE ###
  # Stage 3 (≈4 lines)
    convolutional_block(3L, filters = c(128L, 128L, 512L), stage = 3L, 
                        block='a', s = 2L) %>% 
    identity_block(3L, c(128L, 128L, 512L), stage=3L, block='b') %>% 
    identity_block(3L, c(128L, 128L, 512L), stage=3L, block='c') %>% 
    identity_block(3L, c(128L, 128L, 512L), stage=3L, block='d') %>% 
  
  # Stage 4 (≈6 lines)
    convolutional_block(3L, filters = c(256L, 256L, 1024L), stage = 4L, 
                        block='a', s = 2L) %>% 
    identity_block(3L, c(256L, 256L, 1024L), stage=4L, block='b') %>% 
    identity_block(3L, c(256L, 256L, 1024L), stage=4L, block='c') %>% 
    identity_block(3L, c(256L, 256L, 1024L), stage=4L, block='d') %>% 
    identity_block(3L, c(256L, 256L, 1024L), stage=4L, block='e') %>% 
    identity_block(3L, c(256L, 256L, 1024L), stage=4L, block='f') %>% 
  
  # Stage 5 (≈3 lines)
    convolutional_block(3L, filters = c(512L, 512L, 2048L), stage = 5L, 
                        block='a', s = 2L) %>% 
    identity_block(3L, c(512L, 512L, 2048L), stage=5L, block='b') %>% 
    identity_block(3L, c(512L, 512L, 2048L), stage=5L, block='c') %>% 
  
  # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    layer_average_pooling_2d(c(2L,2L), name = 'avg_pool') %>% 
  
  # output layer
    layer_flatten() %>% 
    layer_dense(classes, activation='softmax', 
                name= paste0('fc', as.character(classes), 
                             kernel_initializer = "glorot_uniform"))
  
  
  # Create model
  model <- keras_model(inputs = X_input, outputs = X)
  
  return(model)
}
#check <- keras_model_sequential()

model <- resnet50(input_shape = c(64L, 64L, 3L), classes = 6L)
summary(model)
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

X_train_orig <- load_dataset()$train_set_x_orig
Y_train_orig <- load_dataset()$train_set_y_orig
X_test_orig <- load_dataset()$test_set_x_orig
Y_test_orig <- load_dataset()$test_set_y_orig
classes <- load_dataset()$classes
train_dataset <- h5file('train_signs.h5', "r")

x_train <- X_train_orig / 255
y_train <- Y_train_orig / 255

x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

history <- model %>% fit(
  x_train, y_train, 
  epochs = 30L, batch_size = 128L, 
  validation_split = 0.2
)

plot(history)

model %>% evaluate(x_test, y_test)
model %>% predict_classes(x_test)


library(h5)
h5file('test_signs.h5', 'r') #in linux, need to run without the 'r' first
h5file('train_signs.h5', 'r')
model <-h5file('ResNet50.h5', 'r') 
  
  load_model_hdf5('ResNet50.h5')
summary(model)

model <- load_model_hdf5("~/signs/ResNet50.h5")
