# Using AWS to run Jupyter Notebook

If you are not allowed to create ML environment on your local computer, you can leverage AWS ec2 instances.

The recommended approach is:
1) create t2.micro ec2 instance and setup ML environment

2) create and test out your model for syntax and valid function with limted epoch runs

3) create a GPU instance ( 1 GPU) ec2 instance and configure it to be accessible for your local machine

4) test out your model with slightly longer run of epochs. BTW, the AWS GPU instance does not provide the optimal GPU performance you would expect from a locally installed NVidia GPU on your desktop

[See how to setup t2.micro instance](setup_micro.ipynb)