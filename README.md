# ef-faas
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

| Test  | Server | Client|  Pi 3 | Pi 4  |
|---|---|---|---|---|
|Download   | 129  |  134 | 130  |  48 |
|Train   | 138  | 143  | 223  | 89  |
|Upload   | 0.176  | <1  | .09  | .02  |
|Predict   | 1  | 6  |  .12 | .08  |