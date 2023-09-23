# Quantum-Neural-Network
- A project that is hosted using **FastAPI** and **Jinja2**, containerized using **Docker** and made as an infrastructure as code using **Terraform** and allows for continuous monitoring using **Prometheus**. 
- The project uses a **Neural Network** whose backend is powered using **quantum computations** and properties to speed up computation. 
- It takes in a csv file with three columns (two features and 1 target) and shows the plot of points along with wrong predictions and time/accuracy comparisons to a normal neural network made using Pytorch


# Usage:

## Full AWS and Terraform
1. Go to config.tf and change region, name for modules and any other necessary configurations
2. Go to AWS ECR in your repository and follow 'view push commands' instructions (if you get an error at first step: view https://stackoverflow.com/questions/60807697/docker-login-error-storing-credentials-the-stub-received-bad-data)
3. Go to AWS load balancer and get DNS name and enter it into your searchbar to see your application

## Docker only
1. Just run `docker-compose up --build` and it should be running on `localhost:8000`
