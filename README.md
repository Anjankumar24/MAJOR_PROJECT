TurboBlade: AI-Driven Compressor Blade Design Optimization

Institution: ACS College of Engineering

Department: Department of Aerospace Engineering

Project: Compressor Blade Design Optimization using Artificial Intelligence

Team Members: Anjan Kumar N, Prerana DS, Lavani C, Tejaswini H

1. Project Overview

Basically, this project is about designing High-Pressure Compressor (HPC) rotor blades automatically using Artificial Intelligence. We are using something called a Surrogate-Based Optimization Framework.

Usually, designing these blades takes a lot of time because we have to run heavy CFD (Computational Fluid Dynamics) simulations again and again. It can take hours just for one design iteration. So, our solution is to use a Deep Neural Network, which acts as a "Surrogate Model" to predict the aerodynamic performance in milliseconds. We have combined this with a Genetic Algorithm (Differential Evolution) to find the best blade shape automatically.

Core Methodology

Surrogate Modeling: We trained a Multi-Layer Perceptron (MLP) using the NREL Aerodynamic Database (which has around 50,000 data points). This helps the model learn the complex relationship between the airfoil shape and its lift/drag coefficients.

Cascade Physics: Since compressor blades are placed close to each other, we applied Weinig’s Lattice Coefficient to correct the isolated airfoil data. This accounts for the Cascade Effects (interference between blades), making our results valid for actual turbomachinery.

Optimization: We used a Differential Evolution algorithm that keeps changing the blade profile to maximize the Lift-to-Drag (L/D) Ratio. It considers specific operating conditions like Reynolds Number and Solidity.

2. System Architecture

The whole system works in three main phases: Training, Optimization, and Generation.

Figure 1: Detailed System Block Diagram illustrating the Nested Optimization Loop.

Input Phase: First, the user gives the flight physics inputs ($Re$, $\alpha$, Solidity) and the geometric constraints (Span, Chord, Twist) through the Dashboard.

Optimization Core: Then, the Genetic Algorithm creates random shapes. The Surrogate Model predicts how good they are, and the Physics Correction Module applies the cascade factors and a skin friction floor to make sure the values are realistic.

3D Generation: Finally, the best 2D profile is stacked, tapered, and twisted to create a proper 3D point cloud that can be directly imported into CATIA V5 and Ansys Fluent.

3. Mathematical Framework

The "AI Brain" of our project is based on these mathematical equations:

Figure 2: Governing Equations for the Neural Network, Cascade Physics, and Optimization Function.

A. The Surrogate Model (ANN)

The relationship between Geometry ($G$) and Aerodynamics ($C_l, C_d$) is approximated by an MLP Regressor:

$$y = f(W_n \cdot ... f(W_1 \cdot x + b_1) ... + b_n)$$

Here, $f(z) = max(0, z)$ is the Rectified Linear Unit (ReLU) activation function, which helps the model understand non-linear aerodynamic behaviors.

B. Cascade Correction (Solidity)

To adapt the isolated airfoil data for a compressor row, we used a correction factor based on Solidity ($\sigma$) (which is the Chord/Spacing ratio):

$$C_{l,cascade} = C_{l,isolated} \times \frac{2}{\pi \sigma}$$

This ensures that our optimized blade considers the blockage and interference effects that happen in a real High-Pressure Compressor stage.

4. Validation & Performance Analysis

We validated our Surrogate Model against a test set of CFD data that the model had never seen before. The results show that the predictions are quite accurate with very less error.

Figure 3: Parity Plot comparing AI Predictions vs. Ground Truth (CFD).

Accuracy: The model got an $R^2$ score of >0.98 for both Lift and Drag coefficients.

Residuals: The error distribution looks like a Gaussian curve centered at zero, which means there is no systematic bias in the model.

Efficiency Gain: The optimizer is consistently finding blade profiles that have 30-40% higher efficiency compared to the average dataset.

5. How to Run the Project

Prerequisites

Make sure you have Python 3.8+ installed. Then, install the required libraries:

pip install -r requirements.txt


Launching the Dashboard

To start the AI Design Tool, just run this command in your terminal:

streamlit run src/app.py


Importing to CATIA

Run the optimization in the dashboard.

Download the optimized_compressor_blade.csv file.

Open the CSV in Excel and delete the first row (headers). Save it as .xls.

Open CATIA V5 -> Generative Shape Design -> Import Points.

Use the Multi-Sections Surface tool to create the blade surface.

6. Future Scope

In the future, we can try to include 3D Flow Features (like Tip Leakage Vortex modeling) directly into the loss function. We can also use Transfer Learning on the NASA E³ (Energy Efficient Engine) datasets to get even better accuracy at transonic speeds.
