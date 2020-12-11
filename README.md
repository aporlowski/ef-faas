# Benchmarking Google Cloud Function for AI Facial Recognition Service

NOTE:
> This document is maintained at:
>
> * <https://github.com/aporlowski/ef-faas/blob/main/README.md>
>

# 1. Introduction

# 2. Results

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



# 3. Appendix

EigenfacesSVM example as FAAS on GCP

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
