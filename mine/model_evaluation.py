import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model.h5')

# Load the test data
test_data = np.load('test_data.npy')
test_labels = np.load('test_labels.npy')

# Preprocess the test data
test_images = []
for image in test_data:
    img = np.expand_dims(image, axis=0)
    img = img.astype('float32') / 255
    test_images.append(img)
test_images = np.array(test_images)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

# Generate the predictions for the test data
test_preds = model.predict_classes(test_images)

# Generate the confusion matrix
cm = confusion_matrix(test_labels, test_preds)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(test_labels)))
plt.xticks(tick_marks, np.unique(test_labels))
plt.yticks(tick_marks, np.unique(test_labels))
fmt = 'd'
thresh = cm.max() / 2.
plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter(fmt))
plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter(fmt))
plt.xlabel('Predicted label')
plt.ylabel('True label')
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylim(len(np.unique(test_labels)) - 0.5, -0.5)
plt.savefig('confusion_matrix.png')
plt.show()