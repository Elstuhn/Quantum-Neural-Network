# Quantum-Neural-Network
- A project that is powered using **FastAPI** and **Jinja2**, hosted using **AWS ECR and ECS**, containerized using **Docker** and made as an infrastructure as code using **Terraform** and allows for **continuous monitoring** using **Prometheus**. 
- The project uses a **Neural Network** whose backend is powered using **quantum computations** and properties to speed up computation. 
- It takes in a csv file with three columns (two features and 1 target) and shows the plot of points along with wrong predictions and time/accuracy comparisons to a normal neural network made using Pytorch
- PS: Terraform sets up all the AWS networking modules like load balancers, elastic IPs, etc.

# Usage:

## Full AWS and Terraform
1. Download prometheus and copy all files/folders except for prometheus.yml into the prometheus folder in this repository
2. Go to config.tf and change region, name for modules and any other necessary configurations
3. Download aws-cli and sign in with credentials (access key id and secret key etc)
4. Go to AWS ECR in your repository and follow 'view push commands' instructions (if you get an error at first step: view https://stackoverflow.com/questions/60807697/docker-login-error-storing-credentials-the-stub-received-bad-data)
5. Go to AWS load balancer and get DNS name and enter it into your searchbar to see your application

## Docker only
1. Download prometheus and copy all files/folders except for prometheus.yml into the prometheus folder in this repository
2. Just run `docker-compose up --build` and it should be running on `localhost:8000`
3. Prometheus is running on `localhost:9090`


# Application (Showcase)
Homepage<br>


Result Page

Ran using OR gates random input and output
<br>
**PS: Reason why the Neural Network in this case runs faster than the QNN might be because of the difference in complexity of architecture. The Neural Network in this case is using a simple Wide Neural Network**
