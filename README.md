# Customer-Churn-Prediction-with-AWS

## Goal:-

Our main goal is to predict the churn rate from mobile phone company based on customer attributes.<br> 
If the provider knows that a customer is thinking of leaving, it can offer timely incentives - such as a phone upgrade or perhaps having a new feature activated – and the customer may stick around. Incentives are often much more cost-effective than losing and reacquiring a customer.


    churn = pd.read_csv("./churn.txt")
    pd.set_option("display.max_columns", 500)
    churn
<img width="561" alt="image" src="https://user-images.githubusercontent.com/69419106/201847677-72e01305-5a34-41b4-a149-c8c65f3a6726.png">


**Explore the data:** <br>
We can see immediately that: - State appears to be quite evenly distributed. - Phone takes on too many unique values to be of any practical use. It’s possible that parsing out the prefix could have some value, but without more context on how these are allocated, we should avoid using it. - Most of the numeric features are surprisingly nicely distributed, with many showing bell-like gaussianity. VMail Message is a notable exception (and Area Code showing up as a feature we should convert to non-numeric).

    churn = churn.drop("Phone", axis=1)
    churn["Area Code"] = churn["Area Code"].astype(object)
    
 Next We’ll look at the relationship between each of the features and our target variable.
 
    for column in churn.select_dtypes(include=["object"]).columns:
    if column != "Churn?":
        display(pd.crosstab(index=churn[column], columns=churn["Churn?"], normalize="columns"))

    for column in churn.select_dtypes(exclude=["object"]).columns:
        print(column)
        hist = churn[[column, "Churn?"]].hist(by="Churn?", bins=30)
        plt.show()
        
    display(churn.corr())
    pd.plotting.scatter_matrix(churn, figsize=(12, 12))
    plt.show()
    
<img width="645" alt="image" src="https://user-images.githubusercontent.com/69419106/201848998-83427631-f97c-4f1a-bf5f-4a07a07f11da.png">
    
Let’s remove one feature from each of the highly correlated pairs: Day Charge from the pair with Day Mins, Night Charge from the pair with Night Mins, Intl Charge from the pair with Intl Mins

let’s convert our categorical features into numeric features.

    churn = churn.drop(["Day Charge", "Eve Charge", "Night Charge", "Intl Charge"], axis=1)
    
    model_data = pd.get_dummies(churn)
    model_data = pd.concat(
        [model_data["Churn?_True."], model_data.drop(["Churn?_False.", "Churn?_True."], axis=1)], axis=1
    )
    
And now let’s split the data into training, validation, and test sets. This will help prevent us from overfitting the model, and allow us to test the model’s accuracy on data it hasn’t already seen.

    train_data, validation_data, test_data = np.split(
    model_data.sample(frac=1, random_state=1729),
    [int(0.7 * len(model_data)), int(0.9 * len(model_data))],
    )
    train_data.to_csv("train.csv", header=False, index=False)
    validation_data.to_csv("validation.csv", header=False, index=False)
    
Now we’ll upload these data to S3 Bucket.

     boto3.Session().resource("s3").Bucket(bucket).Object(
    os.path.join(prefix, "train/train.csv")
    ).upload_file("train.csv")
    boto3.Session().resource("s3").Bucket(bucket).Object(
        os.path.join(prefix, "validation/validation.csv")
    ).upload_file("validation.csv")

## Model Traning:

Moving onto training, first we’ll need to specify the locations of the XGBoost algorithm containers.
    
    container = sagemaker.image_uris.retrieve("xgboost", sess.boto_region_name, "1.5-1")
    display(container)
    
Then, because we’re training with the CSV file format, we’ll create TrainingInputs that our training function can use as a pointer to the files in S3.
    
    s3_input_train = TrainingInput(
        s3_data="s3://{}/{}/train".format(bucket, prefix), content_type="csv"
    )
    s3_input_validation = TrainingInput(
        s3_data="s3://{}/{}/validation/".format(bucket, prefix), content_type="csv"
    )
    
Now, we will specify a few parameters like what type of training instances we’d like to use and how many, as well as our XGBoost hyperparameters. A few key hyperparameters are: - max_depth controls how deep each tree within the algorithm can be built. Deeper trees can lead to better fit, but are more computationally expensive and can lead to overfitting. There is typically some trade-off in model performance that needs to be explored between numerous shallow trees and a smaller number of deeper trees. - subsample controls sampling of the training data. This technique can help reduce overfitting.

    sess = sagemaker.Session()

    xgb = sagemaker.estimator.Estimator(
        container,
        role,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        output_path="s3://{}/{}/output".format(bucket, prefix),
        sagemaker_session=sess,
    )
    xgb.set_hyperparameters(
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.8,
        verbosity=0,
        objective="binary:logistic",
        num_round=100,
    )

    xgb.fit({"train": s3_input_train, "validation": s3_input_validation})

## Deploy

Now that we’ve trained the algorithm, let’s create a model and deploy it to a hosted endpoint.

    xgb_predictor = xgb.deploy(
    initial_instance_count=1, instance_type="ml.m4.xlarge", serializer=CSVSerializer()
    )
    
## Evalution

Now that we have a hosted endpoint running, we can make real-time predictions from our model very easily, simply by making a http POST request. But first, we’ll need to set up serializers and deserializers for passing our test_data NumPy arrays to the model behind the endpoint.

    def predict(data, rows=500):
      split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
      predictions = ""
      for array in data:
          predictions = ",".join([predictions, xgb_predictor.predict(array).decode("utf-8")])

      return np.fromstring(predictions[1:], sep=",")


    predictions = predict(test_data.to_numpy()[:, 1:])
    
## Compare the performance of a machine learning model.

    cm = pd.crosstab(index=test_data['Churn?_True.'], columns=np.round(predictions), rownames=['Observed'], colnames=['Predicted'])
    tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p = (tp+tn)/(tp+tn+fp+fn)*100
    print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
    print("{0:<20}{1:<12}{2:>0}".format("Predicted", "0", "1"))
    print("Observed")
    print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("0", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
    print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("1", fn/(tn+fn)*100,fn, tp/(tp+fp)*100, tp))
    
## Clean up

In this step, you terminate the resources you used in this lab.
Important: Terminating resources that are not actively being used reduces costs and is a best practice. Not terminating your resources will result in charges to your account.

•	Delete your End point ,training artifacts and S3 bucket: In your Jupyter notebook, copy and paste the following code and choose Run.
<img width="525" alt="image" src="https://user-images.githubusercontent.com/69419106/201851629-84140870-b7a3-432d-80fb-4ebb5434834c.png">
<br>
•	Delete your SageMaker Notebook: Stop and delete your SageMaker Notebook.

1.	Open the SageMaker console.
2.	Under Notebooks, choose Notebook instances.
3.	Choose the notebook instance that you created for this tutorial, then choose Actions, Stop. The notebook instance takes up to several minutes to stop. When Status changes to Stopped, move on to the next step.
4.	Choose Actions, then Delete.
5.	Choose Delete.

