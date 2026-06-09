🧠 MobileNetV2 + Custom Loss Function on CIFAR-10
Lightweight Image Classification with Semantic Penalty Loss

A deep learning project that benchmarks MobileNetV2 on the CIFAR-10 dataset under two training regimes: a standard cross-entropy loss and a custom-designed semantic similarity loss that penalises semantically distant misclassifications more heavily.

📊 Key Results
Configuration	Test Accuracy	Final Test Loss
Standard Cross-Entropy (Built-in)	46.81%	5.7145
Custom Similarity Loss	70.71%	1.7712
The custom loss delivers a +23.9 percentage point improvement in test accuracy over the standard baseline.

📁 Project Structure
text
📦 MobileNetV2-CIFAR10/
 ┣ 📓 Untitled6-3.ipynb       ← Main notebook (all code, training, evaluation)
 ┣ 📊 results/
 ┃  ┣ accuracy_comparison.png
 ┃  ┣ loss_comparison.png
 ┃  ┣ final_accuracy_bar.png
 ┃  ┗ penalty_matrix.png
 ┣ 📄 README.md
 ┣ 📄 requirements.txt
 ┗ 📄 .gitignore
📦 Dataset — CIFAR-10
CIFAR-10 contains 60,000 colour images (32×32 pixels) across 10 balanced classes:

Category	Classes
🚗 Vehicles	Airplane, Automobile, Ship, Truck
🐾 Animals	Bird, Cat, Deer, Dog, Frog, Horse
Training set: 50,000 images

Test set: 10,000 images

Preprocessing: Pixel values normalised to 
, labels one-hot encoded

🏗️ Model Architecture
text
Input (32×32×3)
     ↓
MobileNetV2 Base (weights=None, include_top=False)
     ↓
GlobalAveragePooling2D
     ↓
Dense(128, activation='relu')
     ↓
Dropout(0.3)
     ↓
Dense(10, activation='softmax')
Why MobileNetV2?
It uses depthwise separable convolutions to stay computationally lightweight — ideal for resource-constrained environments — while still learning rich feature representations.

🔴 Custom Similarity Loss
The core innovation of this project. Instead of treating all misclassifications equally, the custom loss applies a domain-aware penalty based on how semantically different the predicted class is from the true class.

Penalty Matrix Design:

Same class: 0 (no penalty)

Within same category (vehicle↔vehicle or animal↔animal): 1.0

Cross-category (vehicle↔animal): 3.0 — penalised 3× more

The penalty matrix is shown visually in the results/ folder.

How the loss works:

Compute SparseCategoricalCrossentropy between predictions and true labels

Look up the penalty for each prediction using tf.gather on the penalty matrix

Multiply the cross-entropy by the penalty weight

Return the weighted total loss — errors on semantically distant classes hurt more

This guides the model to avoid the most harmful types of mistakes (e.g., calling a cat a truck), even if it occasionally confuses a cat with a dog.

⚙️ Training Configuration
Parameter	Value
Epochs	10
Batch size	64
Validation split	10%
Optimiser	Adam
Loss (baseline)	Categorical Cross-Entropy
Loss (custom)	Similarity Loss (penalty-weighted CE)
📈 Training Results
Accuracy per Epoch
Epoch	Built-in Val Acc	Custom Val Acc
1	20.48%	39.12%
2	26.43%	53.88%
3	29.44%	59.52%
4	31.48%	62.44%
5	33.03%	64.65%
6	34.97%	66.38%
7	36.30%	67.62%
8	37.31%	68.94%
9	38.15%	69.91%
10	46.81%	70.71%
Charts for accuracy and loss curves are included in the results/ folder.

🔍 Inference Demo
The notebook includes a prediction demo that:

Loads a test image

Runs inference through the trained model

Prints the predicted class, confidence %, and actual class

Displays the image with a green title if correct, red if wrong

👥 Team Contributions
🔵 Himanshu — Custom Loss Function & Model Architecture
Designed the Custom Similarity Loss (similarity_loss): built a domain-aware penalty matrix that penalises cross-category misclassifications (vehicle vs. animal), incorporated it using tf.constant and tf.gather, and combined it with SparseCategoricalCrossentropy for a semantically informed total loss

Defined the 10×10 Penalty Matrix: manually constructed with higher penalties (3.0) for vehicle-animal confusion (classes 0, 1, 8, 9 vs. 2–7), making the model penalise semantically distant errors more than nearby ones

Integrated Base Model Architecture: loaded and configured MobileNetV2 with include_top=False and weights=None for CIFAR-10's 32×32 input, and stacked it with GlobalAveragePooling2D, Dense(128, relu), Dropout(0.3), and a final Softmax classification head

🟢 Vansh — Mobile Backbone & Model Compilation
Integrated MobileNetV2 as Backbone: configured the lightweight MobileNetV2 architecture specifically for the CIFAR-10 32×32×3 input shape, enabling an efficient mobile-friendly feature extractor

Model Compilation with Built-in Loss: compiled the model using the Adam optimiser with categorical_crossentropy loss and accuracy metric — enabling baseline benchmarking against the custom loss variant

Comparative Loss Evaluation: ran training under the standard built-in loss function to allow direct performance comparison with the custom similarity loss, documenting validation accuracy improvements across 10 epochs

🟡 Yashraj — Training Pipeline, Full Evaluation & Visualization
Dataset Loading & Preprocessing: loaded the CIFAR-10 dataset (50,000 train / 10,000 test images), normalised pixel values to 
, and applied to_categorical one-hot encoding for labels — ensuring clean, model-ready inputs before any training began

Training Execution & Monitoring: ran the full model training for 10 epochs with batch size 64 and a 10% validation split; tracked training and validation accuracy/loss throughout both loss configurations, identifying that the custom loss consistently outperformed the baseline from epoch 1 onwards

Evaluation & Final Metrics: evaluated both trained models on the held-out test set using model.evaluate, producing the definitive benchmark: 46.81% accuracy with built-in loss vs 70.71% with custom similarity loss — a 23.9-point improvement confirming the value of semantically-aware training

Visualization Suite: plotted training vs. validation accuracy and loss curves using Matplotlib for both model variants, enabling side-by-side visual comparison of convergence behaviour and overfitting trends across all 10 epochs

Inference Demo & Confidence Scoring: built a live prediction demo that loads test images, runs inference, and reports predicted class with confidence score; implemented colour-coded output titles (green = correct prediction, red = incorrect) for at-a-glance accuracy evaluation

Data Quality Verification: displayed a 3×3 grid of sample training images with their true class labels before training, visually confirming dataset integrity and correct label mapping

🛠️ Setup & Usage
bash
# Clone the repo
git clone https://github.com/your-username/MobileNetV2-CIFAR10-CustomLoss.git
cd MobileNetV2-CIFAR10-CustomLoss

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook Untitled6-3.ipynb
📋 Requirements
Python 3.8+

TensorFlow / Keras

NumPy

Matplotlib

🏆 Key Takeaway
The custom similarity loss function — by embedding domain knowledge directly into the training signal — enables the model to learn semantically meaningful boundaries between classes. The result is a 70.71% accuracy on CIFAR-10 using a lightweight MobileNetV2 backbone trained from scratch, versus only 46.81% under standard cross-entropy.

