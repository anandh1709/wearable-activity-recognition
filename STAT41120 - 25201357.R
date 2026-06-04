# STAT41120 - Machine Learning & A.I. Assignment
# Anandh Venkataraman - 25201357

# Load data
load("C:/Users/anand/OneDrive/Desktop/SML & AIML/data_assignment_activity_recognition.RData")

library(keras3)
library(reticulate)
tf <- import("tensorflow")

# Explore data
cat("Training x:", dim(x), "\n")
cat("Training y:", length(y), "\n")
cat("Test x_test:", dim(x_test), "\n")
cat("Test y_test:", length(y_test), "\n")
table(y)

# Flatten for the DNN: reshape (n, 125, 45) -> (n, 5625)
n_train <- dim(x)[1]
n_test  <- dim(x_test)[1]

x_flat <- array_reshape(x, c(n_train, 125 * 45))
x_test_flat <- array_reshape(x_test, c(n_test, 125 * 45))

cat("Flattened training shape:", dim(x_flat), "\n")
cat("Flattened test shape:",     dim(x_test_flat), "\n")

# Standardise features
train_means <- colMeans(x_flat)
train_sds   <- apply(x_flat, 2, sd)
train_sds[train_sds == 0] <- 1

x_flat      <- scale(x_flat, center = train_means, scale = train_sds)
x_test_flat <- scale(x_test_flat, center = train_means, scale = train_sds)

# Encode labels
y_factor   <- as.factor(y)
yt_factor  <- factor(y_test, levels = levels(y_factor))

# Convert factor labels to 0-indexed integers
y_int      <- as.integer(y_factor) - 1L
y_test_int <- as.integer(yt_factor) - 1L

# Count the number of distinct activity classes
num_classes <- length(levels(y_factor))

# One-hot encode: each label becomes a length-19 binary vector
y_onehot      <- to_categorical(y_int, num_classes)
y_test_onehot <- to_categorical(y_test_int, num_classes)

# Class names for labelling predictions
class_names <- levels(y_factor)
cat("Number of classes:", num_classes, "\n")
cat("Classes:", class_names, "\n")

# Create validation split (shared across both models)
set.seed(42)
val_idx <- sample(1:n_train, size = round(0.15 * n_train))

# DNN validation split (flattened)
x_val <- x_flat[val_idx, ]
y_val <- y_onehot[val_idx, ]
x_tr  <- x_flat[-val_idx, ]
y_tr  <- y_onehot[-val_idx, ]

cat("Train size:", nrow(x_tr), "  Val size:", nrow(x_val), "\n")

# Hyperparameter Tuning for DNN
dnn_lrs      <- c(0.001, 0.0005)
dnn_batches  <- c(32, 64)
dnn_dropouts <- c(0.3, 0.4)

dnn_tuning <- data.frame(
  lr = numeric(), batch = numeric(), dropout = numeric(),
  val_acc = numeric(), val_loss = numeric()
)

for (lr in dnn_lrs) {
  for (bs in dnn_batches) {
    for (dr in dnn_dropouts) {
      cat("DNN: lr =", lr, " batch =", bs, " dropout =", dr, "\n")
      set.seed(42)
      tf$random$set_seed(42L)
      
      model_dnn_tune <- keras_model_sequential() %>%
        layer_dense(units = 512, activation = "relu", input_shape = 5625) %>%
        layer_batch_normalization() %>%
        layer_dropout(rate = dr) %>%
        layer_dense(units = 256, activation = "relu") %>%
        layer_batch_normalization() %>%
        layer_dropout(rate = dr) %>%
        layer_dense(units = 128, activation = "relu") %>%
        layer_dropout(rate = max(dr - 0.1, 0.1)) %>%
        layer_dense(units = num_classes, activation = "softmax")
      
      model_dnn_tune %>% compile(
        optimizer = optimizer_adam(learning_rate = lr),
        loss = "categorical_crossentropy",
        metrics = "accuracy"
      )
      
      model_dnn_tune %>% fit(
        x_tr, y_tr,
        epochs = 50,
        batch_size = bs,
        validation_data = list(x_val, y_val),
        callbacks = list(
          callback_early_stopping(monitor = "val_loss", patience = 7,
                                  restore_best_weights = TRUE)
        ),
        verbose = 0
      )
      
      ev <- model_dnn_tune %>% evaluate(x_val, y_val, verbose = 0)
      dnn_tuning <- rbind(dnn_tuning, data.frame(
        lr = lr, batch = bs, dropout = dr,
        val_acc = ev$accuracy, val_loss = ev$loss
      ))
      cat("  Val accuracy:", round(ev$accuracy, 4),
          " Val loss:", round(ev$loss, 4), "\n\n")
    }
  }
}

# DNN tuning results sorted by validation accuracy
dnn_tuning <- dnn_tuning[order(-dnn_tuning$val_acc), ]
print(dnn_tuning)

best_dnn <- dnn_tuning[1, ]
cat("\n=== Best DNN Hyperparameters ===\n")
cat("Learning rate:", best_dnn$lr, "\n")
cat("Batch size:",   best_dnn$batch, "\n")
cat("Dropout:",      best_dnn$dropout, "\n")
cat("Val accuracy:", round(best_dnn$val_acc, 4), "\n")
cat("Val loss:",     round(best_dnn$val_loss, 4), "\n")

# Retrain Final DNN with Best Hyperparameters

set.seed(42)
tf$random$set_seed(42L)

dnn_model <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = 5625) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = best_dnn$dropout) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = best_dnn$dropout) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = max(best_dnn$dropout - 0.1, 0.1)) %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(dnn_model)

dnn_model %>% compile(
  optimizer = optimizer_adam(learning_rate = best_dnn$lr),
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)

dnn_history <- dnn_model %>% fit(
  x_tr, y_tr,
  epochs = 50,
  batch_size = best_dnn$batch,
  validation_data = list(x_val, y_val),
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 7, 
                            restore_best_weights = TRUE)
  ),
  verbose = 1
)

# Plot DNN training history
plot(dnn_history)

# DNN validation performance
val_eval <- dnn_model %>% evaluate(x_val, y_val, verbose = 0)
cat("DNN validation loss:", val_eval$loss, "\n")
cat("DNN validation accuracy:", val_eval$accuracy, "\n")

# Prepare 3D data for CNN
x_cnn <- array(0, dim = dim(x))
x_test_cnn <- array(0, dim = dim(x_test))

for (i in 1:n_train) {
  x_cnn[i, , ] <- x[i, , ]
}
for (i in 1:n_test) {
  x_test_cnn[i, , ] <- x_test[i, , ]
}

# Standardise per-channel 
channel_means <- rep(0, 45)
channel_sds   <- rep(1, 45)
for (ch in 1:45) {
  vals <- as.vector(x_cnn[, , ch])
  channel_means[ch] <- mean(vals)
  channel_sds[ch]   <- sd(vals)
  if (channel_sds[ch] == 0) channel_sds[ch] <- 1
  x_cnn[, , ch]      <- (x_cnn[, , ch] - channel_means[ch]) / channel_sds[ch]
  x_test_cnn[, , ch]  <- (x_test_cnn[, , ch] - channel_means[ch]) / channel_sds[ch]
}

# Use same validation indices as DNN for fair comparison
x_val_cnn <- x_cnn[val_idx, , ]
x_tr_cnn  <- x_cnn[-val_idx, , ]

cat("CNN train shape:", dim(x_tr_cnn), "\n")
cat("CNN val shape:",   dim(x_val_cnn), "\n")

# Hyperparameter Tuning for CNN
cnn_lrs      <- c(0.001, 0.0005)
cnn_batches  <- c(32, 64)
cnn_dropouts <- c(0.2, 0.3)

cnn_tuning <- data.frame(
  lr = numeric(), batch = numeric(), dropout = numeric(),
  val_acc = numeric(), val_loss = numeric()
)

for (lr in cnn_lrs) {
  for (bs in cnn_batches) {
    for (dr in cnn_dropouts) {
      cat("CNN: lr =", lr, " batch =", bs, " dropout =", dr, "\n")
      set.seed(42)
      tf$random$set_seed(42L)
      
      model_cnn_tune <- keras_model_sequential() %>%
        layer_conv_1d(filters = 64, kernel_size = 5, activation = "relu",
                      input_shape = c(125, 45)) %>%
        layer_batch_normalization() %>%
        layer_max_pooling_1d(pool_size = 2) %>%
        layer_conv_1d(filters = 128, kernel_size = 5, activation = "relu") %>%
        layer_batch_normalization() %>%
        layer_max_pooling_1d(pool_size = 2) %>%
        layer_conv_1d(filters = 256, kernel_size = 3, activation = "relu") %>%
        layer_batch_normalization() %>%
        layer_global_average_pooling_1d() %>%
        layer_dropout(rate = dr) %>%
        layer_dense(units = 128, activation = "relu") %>%
        layer_dropout(rate = max(dr - 0.1, 0.1)) %>%
        layer_dense(units = num_classes, activation = "softmax")
      
      model_cnn_tune %>% compile(
        optimizer = optimizer_adam(learning_rate = lr),
        loss = "categorical_crossentropy",
        metrics = "accuracy"
      )
      
      model_cnn_tune %>% fit(
        x_tr_cnn, y_tr,
        epochs = 50,
        batch_size = bs,
        validation_data = list(x_val_cnn, y_val),
        callbacks = list(
          callback_early_stopping(monitor = "val_loss", patience = 7,
                                  restore_best_weights = TRUE)
        ),
        verbose = 0
      )
      
      ev <- model_cnn_tune %>% evaluate(x_val_cnn, y_val, verbose = 0)
      cnn_tuning <- rbind(cnn_tuning, data.frame(
        lr = lr, batch = bs, dropout = dr,
        val_acc = ev$accuracy, val_loss = ev$loss
      ))
      cat("  Val accuracy:", round(ev$accuracy, 4),
          " Val loss:", round(ev$loss, 4), "\n\n")
    }
  }
}

# CNN tuning results sorted by validation accuracy
cnn_tuning <- cnn_tuning[order(-cnn_tuning$val_acc), ]
print(cnn_tuning)

best_cnn <- cnn_tuning[1, ]
cat("\n=== Best CNN Hyperparameters ===\n")
cat("Learning rate:", best_cnn$lr, "\n")
cat("Batch size:",   best_cnn$batch, "\n")
cat("Dropout:",      best_cnn$dropout, "\n")
cat("Val accuracy:", round(best_cnn$val_acc, 4), "\n")
cat("Val loss:",     round(best_cnn$val_loss, 4), "\n")

# Retraining Final CNN with Best Hyperparameters
set.seed(42)
tf$random$set_seed(42L)

cnn_model <- keras_model_sequential() %>%
  layer_conv_1d(filters = 64, kernel_size = 5, activation = "relu",
                input_shape = c(125, 45)) %>%
  layer_batch_normalization() %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filters = 128, kernel_size = 5, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filters = 256, kernel_size = 3, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_global_average_pooling_1d() %>%
  layer_dropout(rate = best_cnn$dropout) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = max(best_cnn$dropout - 0.1, 0.1)) %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(cnn_model)

cnn_model %>% compile(
  optimizer = optimizer_adam(learning_rate = best_cnn$lr),
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)

cnn_history <- cnn_model %>% fit(
  x_tr_cnn, y_tr,
  epochs = 50,
  batch_size = best_cnn$batch,
  validation_data = list(x_val_cnn, y_val),
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 7,
                            restore_best_weights = TRUE)
  ),
  verbose = 1
)

# Plot CNN training history
plot(cnn_history)

# CNN validation performance
cnn_eval <- cnn_model %>% evaluate(x_val_cnn, y_val, verbose = 0)
cat("CNN validation loss:", cnn_eval$loss, "\n")
cat("CNN validation accuracy:", cnn_eval$accuracy, "\n")

# Model Comparison and Test Evaluation
# Compare validation performance
cat("\n=== Model Comparison (Validation Set) ===\n")
cat("DNN:  accuracy =", round(val_eval$accuracy, 4), 
    " loss =", round(val_eval$loss, 4), "\n")
cat("CNN:  accuracy =", round(cnn_eval$accuracy, 4), 
    " loss =", round(cnn_eval$loss, 4), "\n")
cat("Best model: 1D CNN\n\n")

# Evaluate best model (CNN) on test set
test_eval <- cnn_model %>% evaluate(x_test_cnn, y_test_onehot, verbose = 0)
cat("=== CNN Test Set Performance ===\n")
cat("Test accuracy:", round(test_eval$accuracy, 4), "\n")
cat("Test loss:",     round(test_eval$loss, 4), "\n\n")

# Confusion Matrix
y_pred_probs <- cnn_model %>% predict(x_test_cnn)
y_pred_class <- apply(y_pred_probs, 1, which.max) - 1L

# Convert to factor with class names
y_pred_labels <- factor(class_names[y_pred_class + 1], levels = class_names)
y_true_labels <- factor(class_names[y_test_int + 1],   levels = class_names)

library(caret)
cm <- confusionMatrix(y_pred_labels, y_true_labels)
print(cm)

# Confusion Matrix Heatmap
library(ggplot2)
cm_table <- as.data.frame(cm$table)
colnames(cm_table) <- c("Predicted", "Actual", "Freq")

ggplot(cm_table, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 2.5) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
        axis.text.y = element_text(size = 7)) +
  labs(title = "CNN Confusion Matrix - Test Set")