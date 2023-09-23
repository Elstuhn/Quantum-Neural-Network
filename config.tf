terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "3.0.2"
    }
    aws = {
      source = "hashicorp/aws"
      version = "5.17.0"
    }
  }
}

provider "docker" {
  host = "npipe:////./pipe/docker_engine"
}

# Pulls the image
resource "docker_image" "ubuntu" {
  name = "ubuntu:latest"
}

# Create a container
resource "docker_container" "terraform-docker" {
  image = docker_image.ubuntu.image_id
  name  = "terraform-docker"
  must_run = true
  publish_all_ports = true
  command = [
    "tail",
    "-f",
    "/dev/null"
  ]
}

provider "aws" {
  region = "ap-southeast-1"
}

resource "aws_ecr_repository" "api_ecr" {
  name = "devops-first"
}

module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version="5.1.2"

  name ="main"
  cidr="10.0.0.0/18"

  azs = ["ap-southeast-1a", "ap-southeast-1b"]
  private_subnets = ["10.0.1.0/24"]
  public_subnets = ["10.0.0.0/24"]
  enable_nat_gateway = true 
}

#defining the default internet gateway
data "aws_internet_gateway" "default" {
  filter {
    name   = "attachment.vpc-id"
    values = [module.vpc.vpc_id]
  }
}

# Defining subnets
module "subnets" {
  source              = "git::https://github.com/cloudposse/terraform-aws-dynamic-subnets.git?ref=tags/2.4.1"
  namespace           = "rdx"
  stage               = "dev"
  name                = "devops-first"
  vpc_id              = module.vpc.vpc_id    #data.aws_vpc.default.id
  igw_id              = [module.vpc.igw_id]  #[data.aws_internet_gateway.default.id]
  ipv4_cidr_block     = ["10.0.2.0/24"]
  availability_zones  = ["ap-southeast-1a", "ap-southeast-1b", "ap-southeast-1c"]
}

module "security_group" {
  source = "terraform-aws-modules/security-group/aws//modules/http-80"

  name                = "devops-first-sg"
  vpc_id              = module.vpc.vpc_id
  ingress_cidr_blocks = ["0.0.0.0/0"]
}

#Defining load balancer

module "alb" {
  source  = "terraform-aws-modules/alb/aws"
  version = "8.7.0"

  name            = "devops-first-alb"
  vpc_id          = module.vpc.vpc_id  #data.aws_vpc.default.id
  subnets        = module.subnets.public_subnet_ids
  security_groups = [module.security_group.security_group_id]

  target_groups = [
    {
      name         = "devops-first-tg"
      name-prefix  = "-pref" 
      backend_protocol = "HTTP"
      target_type="ip"
      backend_port=80
      deregistration_delay=10
      load_balancing_cross_zone_enabled=false 
      health_check = {
        enabled=true 
        interval=30 
        path="/health"
        port="traffic-port"
        timeout=6
        protocol="HTTP"
        matcher="200-399"
      }
    }
  ]

  http_tcp_listeners = [
    {
      port               = 80
      protocol           = "HTTP"
      target_group_index = 0
    }
  ]
}

resource "aws_ecs_cluster" "cluster" {
  name = "devops-first-cluster"
}

module "container_definition" {
  source = "git::https://github.com/cloudposse/terraform-aws-ecs-container-definition.git?ref=tags/0.60.0"

  container_name  = "devops-first-container"
  container_image = ""
  port_mappings   = [
    {
      containerPort = 80
      hostPort      = 80
      protocol      = "tcp"
      network_mode  = "awsvpc"
    }
  ]
}

module "ecs_alb_service_task" {
  source = "git::https://github.com/cloudposse/terraform-aws-ecs-alb-service-task.git?ref=tags/0.71.0"

  namespace                 = "rdx"
  stage                     = "dev"
  name                      = "devops-first"
  container_definition_json = module.container_definition.json_map_encoded_list
  ecs_cluster_arn           = aws_ecs_cluster.cluster.arn
  launch_type               = "FARGATE"
  vpc_id                    = module.vpc.vpc_id
  security_group_ids        = [module.security_group.security_group_id]
  subnet_ids                = module.subnets.public_subnet_ids

  health_check_grace_period_seconds  = 60
  ignore_changes_task_definition     = false

  ecs_load_balancers = [
    {
      target_group_arn = module.alb.target_group_arns[0]
      elb_name         = ""
      container_name   = "devops-first-container"
      container_port   = 80
      target_type      = "ip"
  }]
}