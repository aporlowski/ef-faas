# Benchmarking AI Services on Function-as-a-Service Hosting
[Anthony Orlowski](https://github.com/aporlowski/ef-faas)

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
Becuase the cloud functions are stateless, external storage solutions, such as cloud object storage, are required to store and share state across functions. Because each cloud provider offeres thier own flavor of FaaS and storage solutions, choosing to use a FaaS model for a stateful application limits portability of the service code to other platforms. In the EigenfacesSVM example stateful data included the training data, the model after being trained, and images uploaded for label prediciton. Each of these objects is stored in Google Cloud Storage so the dependent functions can download them when invoked. When trying to port this code to another platform, the delovper would need to learn and reimplement that platform's specific storage API, or pay the higher cost (monetary and network latency) to continue using storage services from an external seperate provider.

In contrast to cloud funcitons, GAS generator can support a more traditional serverful model where state is stored on the local OS file system or a locally deployed database. With the GAS generator service code can be written once and deploye don any platform that supports cloudmesh-openapi and ther services dependencies. In takes a similar amount of time to develop an AI service using a FaaS model as it does to use the GAS generator to deploy an AI service to a wide range of supported platforms. This provides the developer flexibility to migrate thier service to an appropriate platform as needed, where FaaS, without extra effort, limits the use case to one particular cloud provider.

FaaS platforms also come with limited resources compared to the wide arrange of platforms supported by CLoudmesh-Openapi. Google Cloud functions currently limits developers to 2048MB of memory, a 2.4 Ghz equivalent processor, and a maximum of 540 seconds for funciton runtime [pricing]. While this was suitable for our example, it will severly lmint the amount ability of FaaS to be used for more expansive AI models. Google has advertised an incrase to 4096MB of memory with 4.8 Ghz quiavlanet process, but at the time of writing we were unable to succesffuly deplot a fucntion to those target resources.

GAS generator, on the other hand, provides access to a wide range of servers (Windows, MacOS, Linux), IoT (Rasbian OS), and container platforms (Docker, Kubernetes) which will allow AI developers to target a platform best suited for thier needs. 

### Development Environment and Prerequisite Knowledge
Developing an AI service using cloud functions comes with some prerequisite knoweldge including: utilizng speficing REST frameworks (like Flask's request objects), utilizing specific storage APIs (like google.cloud.storage), and, if desired, HTML and other GUI presentation langauges.

In contrast, GAS significantly reduces these specific knowledgerequirements. As previously discussed it supports serverful deployment methods that do not require external storage services. It uses OpenAPI to automatically generate a self documenting API and web application presentation of the service hosted by a Flask web server. The developer simple provides the python function code, and the GAS generator turns it into a web app. This signifacnly increase the ability of AI domain experts to share thier work with minmal effort. 

Developing and debugging a cloud function can be difficult becuas the function has to go through a time consuming deployment process before it can be accessed and logs checked for errors and output. Google CLoud Functions does provide instructions on setting up a local development environment, but this is a more complicated development environment setup than that provided by cloudmesh-openapi [local dev]. In contrast GAS generates sevices can be locally developed directly on the same platfrom they will be hosted on.

### Cost
A main advantage for FaaS is the pricing scheme. The AI service is only charged for the runtime of the function and the long-term storage of any backing data. This provides domain experts a cost efficient way to share thier service, particularly if it is infrequently used. GAS generated services provide pricing flexibility by targeting multiple platforms such as cloud virtual machiens and low cost IoT devices.

### Scaling
FaaS can autoscale based on observed usage. GAS generated services leaves it to the developer to scale the service with further infastructure deployments.

### Infastructure Management
Becuase cloud functions are hosted on cloud provider managed servers, the developer does not need to concern themselves deploying and running infastructure. GAS generated services leaves it to the developer to ensure the platform is managed and secured.

### Code Stability
Because FaaS frameworks are developed and managed by commercial organizations, their code has hte potential upside of being more stable and reliable for longterm use.

## Experiement
We benchmark three functions of the EigenfacesSVM service deployed using FaaS and compare it to the benchmarks from [1].We measure three function runtimes:

- **Train** measures the runtime to train the EigenfacesSVM model and store it in cloud storage for future use
- **Upload** measures the runtime to upload an image and store it in cloud storage for future use
- **Predict** measure the runtime to download an image from cloud storage, load the AI model, and run a prediction on the image

We measure these functions from two different perspectices:

- **Client** This is the function runtime from the remote client
- **Server** This is the funciton runtime as measured directly on the server directly within the funciton

We measure runtimes using the cloudmesh.common.Benchmark utility. In the case of client measurement, we can measure this in our test python program. In the case of server measurement, we run the benchmark locall within the function on the server, and return its results in the HTTP response. We expect the client runtime to be slower than the server runtime to account for both network round-trip-times, and the amount of time it takes to prepare an instance for function execution. 

Because cloud functions are epehmeral we conduct two tests. One in which the majority of instances are cold started, and a second where warm-start instances are already running. We constructed this test by first deploying a new function, ensuring there was only one instance ruunning, and then conducting 30 requets in parallel. The remaining 29 requetes will incur a cold-start situation. Immediately following the completiong of the first test we run an additional 30 requetse to tray and capture warm-start instances. Thus theare two conditions cloud function instances are captured in:

- **Cold-start** A maixmum of 1 instance is running before 30 parallel requests
- **Warm-start** This test of 30 parallel requests runs immedialy upon the competion of the cold-start test

From the perspective of the remote client, we expect the runtime of hte cold-start functions to be signican longer becuas the cloud provider must first prepare a containe rand initialize the function enivornment before it can be run. We expect warm-start function invocations to be significantly faster. From the server side perspective we expect the cold-start and warm-start function times to be similar, as the timer is not running during instance setup.

Finally, we measure two seperate cloud function sizes to see if an increase in resources imrproves performance. Google cloud functions has set resource configurations [pricing]. We determined we had a minimum meory requirement of 1GB, so were only able to test 1GB and 2GB variations. 4GB variations are advertised, but we were not succesffully able to deploy to that target configuration on the us-east1 region at time of writing. Thus there are two resource variants:

- **1gb** Provides 1024MB of memory and a 1.4GHz processor
- **2gb** Provides 2048MB of memory and a 2.4GHz processor

We expect functions to run faster on the variant with greater resources. Interestingly, we identify that these resources are similar in quantity to those used by Rasberry Pi's from [1], and we are curios to see how thier performance measures up.

## Results

In Figure 3 we show the runtime of the train function under the various cloud function conditions. The bars show the average of 30 trials, and the error bars show the standard deviation of the 30 trials. As expected, cold-start functions are significantly slower than warm-start functions. However, the long runtime of this function reduces amoritizes this cold-start cost better than short running functions. Surprisingly, we do not see a distinguishable improvement on functions allocated more resources. We were hoping to deploy to a additinal resource configurations to further study this, but these two configuraitons were the only two with sufficient resources at the time of hte study.

![Train FaaS](https://github.com/aporlowski/ef-faas/raw/main/images/Train_graph.png)

**Figure 3:** Train function runtime for cloud function with various conditions.

In Figure 4 we show the runtime of the upload function under the various cloud function conditions. The bars show the average of 30 trials, and the error bars show the standard deviation of the 30 trials. As expected, cold-start functions are significantly slower than warm-start functions. From the client perspective, the cold-start and warm-start difference is especially large, as it comprises a significant poriton of the overall function runtime. Surprisingly, there is a large difference between the client warm start runtime and the server runtimes. This implies there is additional delay in handling requests besides environment setup, or our experiment did not achienve a perfect warm-instance hit rate. Unlike in Figure 3, in this experiemtn we do see an identifiable decreases in runtime that comes with increased resources. 

![Upload Faas](https://github.com/aporlowski/ef-faas/raw/main/images/Upload_graph.png)

**Figure 4:** Upload function runtime for cloud function with various conditions.

In Figure 5 we show the runtime of the upload function under the various cloud function conditions. The bars show the average of 30 trials, and the error bars show the standard deviation of the 30 trials. This figure provides us further confirmation of the observations made in the discussion of Figure 4.

![Predict Faas](https://github.com/aporlowski/ef-faas/raw/main/images/Predict_graph.png)

**Figure 5:** Predict function runtime for cloud function with various conditions.

In Figure 6 we show the tuntime of the train function compared to the results from [1]. Machine specifications from [1] are shown in Table 2. The bars show the average the rials, and the error bars show the standard deviation of the trials. We only show the server obserced runtimes as client side measurements were not measured in [1]. As predicted we peformance in the range of that measured of the two Rasbperry Pi models, and that traditional virutal machines significantly outperform the FaaS offerings. Considering we used hte highest working resource configuration available, it is surprising that Cloud Functions has such relatively low peformance. Higher resource limits will be required from Google Cloud Functions before larger AI services could consider it a possible deployment target.   

![Train Platforms](https://github.com/aporlowski/ef-faas/raw/main/images/Train_platforms_graph.png)

**Figure 6:** Server-side train function runtime for cloud function compared with other platforms.

In Figure 7 we show the runtime of the upload funciton compared to results from [1]. he bars show the average the rials, and the error bars show the standard deviation of the trials. In [1] the cloud VMs were the only remotely deployed services, so from a client perspective, we can only compare the FaaS to the cloud VMs. In [1] we identified that the network round-trip-time was the dominate componenent of the function runtime. Here we obserce the FaaS functions perform signifacntly worse despite having similar network round-trip-times, as all clouds and FaaS functions were deployed to east cost data centers and access from the same remote network. This graph shows warm-start functions are not a significant enough improvement for AI services that desire to decrease latency.

![Upload Platforms](https://github.com/aporlowski/ef-faas/raw/main/images/Upload_platforms_graph.png)

**Figure 7:** Client-side upload function runtime for cloud function compared with other platforms.

In Figure 8 we show the runtime of the upload funciton compared to results from [1]. The bars show the average the rials, and the error bars show the standard deviation of the trials. This figure provides us further confirmation of the observations made in the discussion of Figure 7.

![Predict Platforms](https://github.com/aporlowski/ef-faas/raw/main/images/Predict_platforms_graph.png)

**Figure 8:** Client-side predict function runtime for cloud function compared with other platforms.

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

## Limitations
This work focuses on generating benchmark results to compare to [1], thus it does not yet implement a fully generealized EigenfacesSVM service. There are some features of the functions that need to be completed for a more generalized service that does more than the Scikit-learn example. As it stands the functions operate on one specific data set for hte donwload function, and one specific image for the predict funciton. The upload function can upload arbitary images. Extending the code to be a full service will require additional argument passing and processing and for the predict and download funciton. Most of the logic is present to finish these features, but it a complete implementation is not the focus of this work. This limits do not detract from the benchmark validity, simply the generalized use of the facial recognition service.

Our warm-start experiement is designed such that recently used containers are available, and while our results show this was the case, we do not explictily measure what percentage of warm-start instances are used. Using global instance state, one can set a flag denoting whether an instance has been previously used. Measuring what percent of requets can find this flag may be a good opportunity for better warm-start measurements. 

A full cost analysis is not presented to identify the true cost efficiency that FaaS model may afford. We identified this is not trivial to measure as cost incurs both storage usage and function invoations which are priced and billed seperately. Pricint sroage further sepeartes data-at-rest charges and data network egres charges.  For a true cost analysis, a robust set of use cases including: amount of data, length of data storage, number of function invocations, and regional distribution of services need be created and compared to a similar serverful deployment. This is outside the scale of this work.

## Conclusion
In this project we  deploy and benchmark an AI service using the Google Cloud Functions function-as-a-servce platform. We study this with the intent to identify if FaaS is a viable and easy-to-use model for AI domain experts to develop and share AI services. We demonstrate that FaaS has the benefit of per-function call billing, autoscaling, and managed infastructure, but that it is limited in performance, response latency, vendor specific and complex development, platform flexibility, and requires pre-requisite knoweldge AI domain experts do not wnat to learn. We compare this to our previous work with the Generalized AI Service (GAS) Generator and show GAS generator overcomes many of these limittions.

## Acknowledgements
We like to thank Grego von Laszesski, Richard Otten, Reilly Markowitz, Sunny Gandhi, Adam Chai, Caleb Wilson, and Geoffry C. Fox for the prior AI service generation and benchmarking that this work is based on. 

## References

[pricing] https://cloud.google.com/functions/pricing
[local dev] https://cloud.google.com/functions/docs/running/overview

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

