# complex_operator.py
from DeepONet.generate_data import generate_lambda_dataset, generate_kernel_dataset
from DeepONet.training import train_operator

# Sample dataset λ(x) = 50cos(γccos^-1(x))
x, lam_dataset, gammas = generate_lambda_dataset(samples=1000, nx=51)
# Sample dataset k(x,y)
k_dataset = generate_kernel_dataset(lam_dataset=lam_dataset)
# Train Operator
Operator = train_operator(x=x, lam_dataset=lam_dataset, k_dataset=k_dataset, samples=1000, nx=51, trunk_layers=[128, 256, 256], iterations=7000)
Operator.save("Operator.pth")