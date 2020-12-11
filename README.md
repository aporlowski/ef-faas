# Benchmarking AI Services with Function-as-a-Service Hosting

NOTE:
> This document is maintained at:
>
> * <https://github.com/aporlowski/ef-faas/blob/main/README.md>
>

## 1. Introduction

In this project we adapt, deploy, and benchmark an AI service using Google’s Cloud Functions, a function-as-a-service (FaaS) platform, and compare the performance to benchmarks conducted in [1] for the same service hosted on cloud virtual machines, IoT devices, and locally hosted containers. In recent work we use the Generalized AI Service (GAS) Generator to autogenerate and deploy RESTful AI services using the cloudmesh-openapi utility [1]. The GAS generator provides AI domain experts a simple interface to share AI services while automating infastrcuture deployment and service hosting with easy to use command line interface provided by Cloudmesh and Cloudmesh-openapi. The example service, EigenfacesSVM, is a facial recoginition example taken from Scikit-learn that is modified to be an AI service [2].  The purpose of this project is to compare benchmark results of the service deployed in a FaaS model to the serverful paradigms tested in the original work, and explain the development and deployment differences between the GAS generator and the FaaS platform.

## Background

### Autogenerated AI services with GAS

In [1] we adapted the EigenfacesSVM model from Scikit-learn to be a set of four Python functions that download image data from a remote repository, trains the AI model, uploads an image for prediction, and predicts the label for the uploaded image. The GAS generator uses Cloudmesh-openapi to automatically translate and deploy these Python functions into a RESTful service. The service is developed once and deployed to any target platform that Cloudmesh-openapi supports. For example, we demonstrated a multi-cloud service in which we deployed and benchmared the AI service on three clouds simultaneously using Cloudmesh and Cloudmesh-opeanpi. We benchmarked the AI functions on cloud virtual machines from AWS, Google, and Azure, as well as Raspberry Pi platforms, a Mac book, and a docker container run on that MacBook. It is important to note that these services were deployed in a serverful manner, where the hosting platform is continuously running the service. In contrast, this projects aims to develop the same servcice in a serverless manner, where each invocation of the service is potentially carried out by differenet instance.

### Serverless Computing and Cloud Functions
Cloud functions are a part of the serverless computing model in which cloud provider's offer managed and autoscaling execution environments for customers to deploy thier code. By providing managed infastructure, cloud providers reduce the demands on developers to deploy and maintain infastructure. In the case of AI domain experts trying to deploy a Cloud functions provide extra fine grained billing where customers are only chrged for the execution of the function. This contrasts traditional VM pricing where customers are charged per hour the machine is on regardles if the encapsulated service is being used or not. Providers accomplish this by standing-up and deploying customer code into a leightweight container on demand. Each deployed container is an instance, and hte cloud provider can scale the number of instances running based on the observed demand. Because cloud function instances are ephemeral, cloud functions are best suited for stateless and imdepotent operations. If state is needed to be saved or shared between instances, then they will have to interface through a storage solution such as cloud object storage. Additionally, cloud functions are not directly addressable, so a client cannot attempt to communicate with a specific instance.  

## Architecture

### Serverful AI Service Hosting using Cloudmesh-Openapi

### Serverless AI Service Hosting on Google Cloud Functions


## Lessons Learned from Development and Deployment

In this section I will disucss advantages and disadvantages of developing and deplouing the AI service with both the GAS generator and FaaS. 

### Platform Flexibility
Becuase the cloud functions are stateless, external storage solutions, such as cloud object storage, are required to store and share state functions. Because each cloud provider offeres thier own flavor of FaaS and storage solutions, choosing to use a FaaS model for a stateful application limits portability of the service code to other platforms. In the EigenfacesSVM example stateful data included the training data, the model after being trained, and images uploaded for label prediciton. Each of these objects is stored in Google Cloud Storage so the dependent functions can download them when invoked. When trying to port this code to another platform, the delovper would need to learn and reimplement that platform's specific storage API, or pay the higher cost (monetary and network latency) to continue using storage services from an external seperate provider.

In contrast to cloud funcitons, GAS generator can support a more traditional serverful model where state is stored on the local OS file system or a locally deployed database. With the GAS generator service code can be written once and deploye don any platform that supports cloudmesh-openapi and ther services dependencies. In takes a similar amount of time to develop an AI service using a FaaS model as it does to use the GAS generator to deploy an AI service to a wide range of supported platforms.



### Development Environment and Prerequisite Knowledge

### Cost

### Scaling

### Infastructure Management


The 
1.vendor specific development
2. rest knowledge
3. cloud storage
4 more complicated development/testing due to having to deploy the function and check logs for errors
5. No auto-generated GUI with self documenting from OpenAPI 
6. limited resources (4GB max, 540s max runtime, no GPU).

1) only chaged per function use + data storage. 2) Auto-scaling  3) I expect codebase to be more stable 4) no infastructure management. Becuase cloud functions are hosted on cloud provider managed servers, the developer does not need to concern themselves deploying and running infastructure. Compared to GAS

## Experiement

## Results

![Train FaaS](https://github.com/aporlowski/ef-faas/raw/main/images/Train_graph.png)

**Figure 1:** Train function runtime for cloud function with various conditions.

![Upload Faas](https://github.com/aporlowski/ef-faas/raw/main/images/Upload_graph.png)

**Figure 2:** Upload function runtime for cloud function with various conditions.

![Predict Faas](https://github.com/aporlowski/ef-faas/raw/main/images/Predict_graph.png)

**Figure 3:** Predict function runtime for cloud function with various conditions.

![Train Platforms](https://github.com/aporlowski/ef-faas/raw/main/images/Train_platforms_graph.png)

**Figure 4:** Server-side train function runtime for cloud function compared with other platforms.

![Upload Platforms](https://github.com/aporlowski/ef-faas/raw/main/images/Upload_platforms_graph.png)

**Figure 5:** Client-side upload function runtime for cloud function compared with other platforms.

![Predict Platforms](https://github.com/aporlowski/ef-faas/raw/main/images/Predict_platforms_graph.png)

**Figure 6:** Client-side predict function runtime for cloud function compared with other platforms.

**Table 1:** Complete test measurements.

| size     | party   | type   | test    |   mean |    min |    max |   std |
|:---------|:--------|:-------|:--------|-------:|-------:|-------:|------:|
| 1gb      | client  | cold   | predict |   7.27 |   2.14 |   9.12 |  0.26 |
| 1gb      | client  | warm   | predict |   3.92 |   0.64 |   6.16 |  0.29 |
| 1gb      | server  | cold   | predict |   0.7  |   0.52 |   0.92 |  0.02 |
| 1gb      | server  | warm   | predict |   0.57 |   0.34 |   1.46 |  0.03 |
| 2gb      | client  | cold   | predict |   7.08 |   1.18 |   8.09 |  0.24 |
| 2gb      | client  | warm   | predict |   3.64 |   0.48 |   5.46 |  0.32 |
| 2gb      | server  | cold   | predict |   0.63 |   0.55 |   0.75 |  0.01 |
| 2gb      | server  | warm   | predict |   0.55 |   0.27 |   0.7  |  0.02 |
| aws      | client  |        | predict |   0.4  |   0.26 |   0.8  |  0.18 |
| azure    | client  |        | predict |   0.36 |   0.24 |   0.6  |  0.13 |
| google   | client  |        | predict |   0.36 |   0.27 |   0.82 |  0.16 |
| 1gb      | client  | cold   | train   | 129.38 | 112.51 | 178.07 |  2.92 |
| 1gb      | client  | warm   | train   | 123.23 |  94.06 | 183.9  |  2.81 |
| 1gb      | server  | cold   | train   | 123.93 | 107.72 | 171.5  |  2.96 |
| 1gb      | server  | warm   | train   | 119.23 |  93.67 | 179.99 |  2.75 |
| 2gb      | client  | cold   | train   | 131.19 | 113.92 | 171.67 |  2.29 |
| 2gb      | client  | warm   | train   | 118.33 |  61.43 | 138.82 |  3.04 |
| 2gb      | server  | cold   | train   | 125.74 | 110.26 | 164.44 |  2.16 |
| 2gb      | server  | warm   | train   | 114.8  |  61.22 | 135.2  |  2.89 |
| aws      | server  |        | train   |  35.72 |  34.91 |  46.5  |  1.73 |
| azure    | server  |        | train   |  40.28 |  35.3  |  47.5  |  3.32 |
| docker   | server  |        | train   |  54.72 |  54.72 |  54.72 |  0    |
| google   | server  |        | train   |  42.04 |  41.52 |  45.93 |  0.71 |
| mac book | server  |        | train   |  33.82 |  33.82 |  33.82 |  0    |
| pi 3b+   | server  |        | train   | 222.61 | 208.56 | 233.48 |  8.4  |
| pi 4     | server  |        | train   |  88.59 |  87.83 |  89.35 |  0.32 |
| 1gb      | client  | cold   | upload  |   5.97 |   1.42 |   7.67 |  0.23 |
| 1gb      | client  | warm   | upload  |   4.18 |   0.34 |   7.05 |  0.38 |
| 1gb      | server  | cold   | upload  |   0.2  |   0.14 |   0.42 |  0.01 |
| 1gb      | server  | warm   | upload  |   0.17 |   0.09 |   0.31 |  0.01 |
| 2gb      | client  | cold   | upload  |   4.96 |   0.9  |   5.97 |  0.16 |
| 2gb      | client  | warm   | upload  |   2.93 |   0.31 |   4.97 |  0.33 |
| 2gb      | server  | cold   | upload  |   0.17 |   0.13 |   0.23 |  0.01 |
| 2gb      | server  | warm   | upload  |   0.13 |   0.08 |   0.16 |  0.01 |
| aws      | client  |        | upload  |   0.43 |   0.16 |   1.13 |  0.21 |
| azure    | client  |        | upload  |   0.32 |   0.15 |   0.5  |  0.15 |
| google   | client  |        | upload  |   0.31 |   0.18 |   0.73 |  0.18 |

## Acknowledgements

## References

## Appendix

### DeployEigenfacesSVM example as FAAS on GCP

```
cd ~/PycharmProjects/ef-faas/service

gcloud functions deploy eigenfaces_download_data_http --set-env-vars USER=benchmark --runtime python38 --trigger-http --allow-unauthenticated --memory=1024MB --timeout=540s --region=us-east1
gcloud functions deploy eigenfaces_train_http --set-env-vars USER=benchmark --runtime python38 --trigger-http --allow-unauthenticated --memory=1024MB --timeout=540s --region=us-east1
gcloud functions deploy eigenfaces_upload_http --set-env-vars USER=benchmark --runtime python38 --trigger-http --allow-unauthenticated --memory=1024MB --timeout=540s --region=us-east1
gcloud functions deploy eigenfaces_predict_http --set-env-vars USER=benchmark --runtime python38 --trigger-http --allow-unauthenticated --memory=1024MB --timeout=540s --region=us-east1

curl https://us-east1-anthony-orlowski.cloudfunctions.net/eigenfaces_download_data_http >> out.txt
curl https://us-east1-anthony-orlowski.cloudfunctions.net/eigenfaces_train_http >> out.txt
curl -F example_image.jpg=@example_image.jpg  https://us-east1-anthony-orlowski.cloudfunctions.net/eigenfaces_upload_http >> out.txt
curl https://us-east1-anthony-orlowski.cloudfunctions.net/eigenfaces_predict_http >> out.txt
```

```
gcloud functions describe eigenfaces_download_data_http

gcloud functions delete eigenfaces_download_data_http
```

