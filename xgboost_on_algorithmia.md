# Sentiment Analysis with XGBoost on Algorithmia

With this notebook, we will be training an XGBoost model on Amazon's Musical Instrument Reviews dataset and be able to use this model to predict the sentiment of the given texts. If you would like to see the final product first, you can check out this algorithm in action at https://algorithmia.com/algorithms/asli/xgboost_basic_sentiment_analysis

## Overview
Let's first go over the steps we will cover in this notebook. We will start with the end in mind and then slowly build up to that point. At the end of this demo, you will have an up and running algorithm on [Algorithmia](https://algorithmia.com), ready to serve its predictions upon your requests!

Step by step, we will: 

1. Create an algorithm on Algorithmia

2. Clone the algorithm's repository on our local machine, so that we develop locally 

3. Create the basic algorithm script and the dependencies file. We will code our script in advance, assuming that our model will be sitting on a remote path on Algorithmia and our script will load the model from there. We will then make these assumptions true!

4. Commit and push these files to Algorithmia and get our Algorithm's container built

5. Load the training data

6. Preprocess the data

7. Setup an XGBoost model and do a mini hyperparameter search

8. Fit the data on our model

9. Get the predictions

10. Check the accuracy

11. Repeat the steps through 6 and 10 until we are happy with our model :)

11. Once we are happy, upload the to Algorithmia and have it up and ready to serve our upcoming prediction requests!

12. Test our published algorithm with sample requests

## Getting up and ready on Algorithmia

Let's first create an algorithm on Algorithmia and then build on it slowly.
After importing the necessary packages, we'll define the variables to use across many of our calls to Algorithmia, through its [Python API](https://algorithmia.com/developers/clients/python).


```python
%load_ext autoreload
%autoreload 2
```


```python
import Algorithmia
import algorithmia_utils
```


```python
api_key = "YOUR_API_KEY"
username = "YOUR_USERNAME"
algo_name = "xgboost_basic_sentiment_analysis"
local_dir = "../algorithmia_repo"

algo_utility = algorithmia_utils.AlgorithmiaUtils(api_key, username, algo_name, local_dir)
```

### Creating the algorithm and cloning its repo
You would only need to do this step once, because you only need one algorithm and cloning it once on your local environment is enough.

We will use the utility functions defined on our imported Algorithmia utility script.


```python
# You would need to call these two functions only once
algo_utility.create_algorithm()
algo_utility.clone_algorithm_repo()
```

### Adding the algorithm script and the dependencies
Let's create the algorithm script that will run when we make our requests and the dependency file that will be used when building the container for our algorithm on the Algorithmia environment.

We will be creating these two files programmatically with the `%%writefile` macro, but you can always use another editor to edit and save them later when you need.


```python
%%writefile $algo_utility.algo_script_path
import Algorithmia
import joblib
import numpy as np
import pandas as pd
import xgboost

model_path = "data://asli/xgboost_demo/musicalreviews_xgb_model.pkl"
client = Algorithmia.client()
model_file = client.file(model_path).getFile().name
loaded_xgb = joblib.load(model_file)

# API calls will begin at the apply() method, with the request body passed as 'input'
# For more details, see algorithmia.com/developers/algorithm-development/languages
def apply(input):
    series_input = pd.Series([input])
    result = loaded_xgb.predict(series_input)
    # Returning the first element of the list, as we'll be taking a single input for our demo purposes
    # As you'll see while building the model: 0->negative, 1->positive
    return {"sentiment": result.tolist()[0]}
```

    Overwriting ../algorithmia_repo/asli_test_utils/src/asli_test_utils.py



```python
%%writefile $algo_utility.dependency_file_path
algorithmia>=1.0.0,<2.0
scikit-learn
pandas
numpy
joblib
xgboost
```

    Overwriting ../algorithmia_repo/asli_test_utils/requirements.txt


### Adding these files to git, commiting and pushing
Now we're ready to upload our changes to our remote repo on Algorithmia and our algorithm will be built on the Algorithmia servers and get ready to accept our requests.


```python
algo_utility.push_algo_script_with_dependencies()
```

    Enumerating objects: 1, done.
    Counting objects: 100% (1/1)
    Counting objects: 100% (1/1), done.
    Writing objects: 100% (1/1)
    Writing objects: 100% (1/1), 189 bytes | 189.00 KiB/s, done.
    Total 1 (delta 0), reused 0 (delta 0)
    remote: Updating references: 100% (1/1)
    remote: Updating references: 100% (1/1)
    remote:
    remote: Build successful for algo://asli/asli_test_utils/5f71559ce69de566a1d9ff9606eb4c6220e72bd0
    remote:


## Building the XGBoost model
Now it's time to build our model!


```python
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler  # for scaling
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from string import punctuation
from nltk.corpus import stopwords

from scipy.stats import uniform

from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import joblib
```

### Load the training data
Let's load our training data, take a look at a few rows and one of the review texts in detail.


```python
data = pd.read_csv("./data/amazon_musical_reviews/Musical_instruments_reviews.csv")
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reviewerID</th>
      <th>asin</th>
      <th>reviewerName</th>
      <th>helpful</th>
      <th>reviewText</th>
      <th>overall</th>
      <th>summary</th>
      <th>unixReviewTime</th>
      <th>reviewTime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A2IBPI20UZIR0U</td>
      <td>1384719342</td>
      <td>cassandra tu "Yeah, well, that's just like, u...</td>
      <td>[0, 0]</td>
      <td>Not much to write about here, but it does exac...</td>
      <td>5.0</td>
      <td>good</td>
      <td>1393545600</td>
      <td>02 28, 2014</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A14VAT5EAX3D9S</td>
      <td>1384719342</td>
      <td>Jake</td>
      <td>[13, 14]</td>
      <td>The product does exactly as it should and is q...</td>
      <td>5.0</td>
      <td>Jake</td>
      <td>1363392000</td>
      <td>03 16, 2013</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A195EZSQDW3E21</td>
      <td>1384719342</td>
      <td>Rick Bennette "Rick Bennette"</td>
      <td>[1, 1]</td>
      <td>The primary job of this device is to block the...</td>
      <td>5.0</td>
      <td>It Does The Job Well</td>
      <td>1377648000</td>
      <td>08 28, 2013</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A2C00NNG1ZQQG2</td>
      <td>1384719342</td>
      <td>RustyBill "Sunday Rocker"</td>
      <td>[0, 0]</td>
      <td>Nice windscreen protects my MXL mic and preven...</td>
      <td>5.0</td>
      <td>GOOD WINDSCREEN FOR THE MONEY</td>
      <td>1392336000</td>
      <td>02 14, 2014</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A94QU4C90B1AX</td>
      <td>1384719342</td>
      <td>SEAN MASLANKA</td>
      <td>[0, 0]</td>
      <td>This pop filter is great. It looks and perform...</td>
      <td>5.0</td>
      <td>No more pops when I record my vocals.</td>
      <td>1392940800</td>
      <td>02 21, 2014</td>
    </tr>
  </tbody>
</table>
</div>




```python
data["reviewText"].iloc[1]
```




    "The product does exactly as it should and is quite affordable.I did not realized it was double screened until it arrived, so it was even better than I had expected.As an added bonus, one of the screens carries a small hint of the smell of an old grape candy I used to buy, so for reminiscent's sake, I cannot stop putting the pop filter next to my nose and smelling it after recording. :DIf you needed a pop filter, this will work just as well as the expensive ones, and it may even come with a pleasing aroma like mine did!Buy this product! :]"



### Preprocessing
Time to process our texts! Basically, we'll:
- Remove the English stopwords
- Remove punctuations
- Drop unused columns


```python
def threshold_ratings(data):
    def threshold_overall_rating(rating):
        return 0 if int(rating)<=3 else 1
    data["overall"] = data["overall"].apply(threshold_overall_rating)

def remove_stopwords_punctuation(data):
    data["review"] = data["reviewText"] + data["summary"]

    puncs = list(punctuation)
    stops = stopwords.words("english")

    def remove_stopwords_in_str(input_str):
        filtered = [char for char in str(input_str).split() if char not in stops]
        return ' '.join(filtered)

    def remove_punc_in_str(input_str):
        filtered = [char for char in input_str if char not in puncs]
        return ''.join(filtered)

    def remove_stopwords_in_series(input_series):
        text_clean = []
        for i in range(len(input_series)):
            text_clean.append(remove_stopwords_in_str(input_series[i]))
        return text_clean

    def remove_punc_in_series(input_series):
        text_clean = []
        for i in range(len(input_series)):
            text_clean.append(remove_punc_in_str(input_series[i]))
        return text_clean

    data["review"] = remove_stopwords_in_series(data["review"].str.lower())
    data["review"] = remove_punc_in_series(data["review"].str.lower())

def drop_unused_colums(data):
    data.drop(['reviewerID', 'asin', 'reviewerName', 'helpful', 'unixReviewTime', 'reviewTime', "reviewText", "summary"], axis=1, inplace=True)

def preprocess_reviews(data):
    remove_stopwords_punctuation(data)
    threshold_ratings(data)
    drop_unused_colums(data)
```


```python
preprocess_reviews(data)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>overall</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>much write here exactly supposed to filters po...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>product exactly quite affordablei realized dou...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>primary job device block breath would otherwis...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>nice windscreen protects mxl mic prevents pops...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>pop filter great looks performs like studio fi...</td>
    </tr>
  </tbody>
</table>
</div>



### Split our training and test sets


```python
rand_seed = 42
X = data["review"]
y = data["overall"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_seed)
```

### Mini randomized search
Let's set up a very basic cross-validated randomized search over parameter settings.


```python
params = {"max_depth": range(9,12), "min_child_weight": range(5,8)}
rand_search_cv = RandomizedSearchCV(XGBClassifier(), param_distributions=params, n_iter=5)
```

### Pipeline to vectorize, transform and fit
Time to vectorize our data, transform it and then fit our model to it.
To be able to feed the text data as numeric values to our model, we will first convert our texts into a matrix of token counts using a CountVectorizer. Then we will convert the count matrix to a normalized tf-idf (term-frequency times inverse document-frequency) representation. Using this transformer, we will be scaling down the impact of tokens that occur very frequently, because they convey less information to us. On the contrary, we will be scaling up the impact of the tokens that occur in a small fraction of the training data because they are more informative to us.


```python
model  = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', rand_search_cv)
])
model.fit(X_train, y_train)
```




    Pipeline(steps=[('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('model',
                     RandomizedSearchCV(estimator=XGBClassifier(base_score=None,
                                                                booster=None,
                                                                colsample_bylevel=None,
                                                                colsample_bynode=None,
                                                                colsample_bytree=None,
                                                                gamma=None,
                                                                gpu_id=None,
                                                                importance_type='gain',
                                                                interaction_constraints=None,
                                                                learning_rate=None,
                                                                max_delta_step=None,
                                                                max_depth=None,
                                                                min_child_weight=None,
                                                                missing=nan,
                                                                monotone_constraints=None,
                                                                n_estimators=100,
                                                                n_jobs=None,
                                                                num_parallel_tree=None,
                                                                random_state=None,
                                                                reg_alpha=None,
                                                                reg_lambda=None,
                                                                scale_pos_weight=None,
                                                                subsample=None,
                                                                tree_method=None,
                                                                validate_parameters=None,
                                                                verbosity=None),
                                        n_iter=5,
                                        param_distributions={'max_depth': range(9, 12),
                                                             'min_child_weight': range(5, 8)}))])



### Predict and calculate accuracy


```python
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {round(acc * 100, 2)}")
```

    Model Accuracy: 89.14


### Save the model
Once we're happy with our model's accuracy, let's save it locally first and then take it from there and upload to Algorithmia.
For the Algorithmia upload, we will use our previously defined function.


```python
model_name = "musicalreviews_xgb_model.pkl"
local_path = f"model/{model_name}"
```


```python
joblib.dump(model, local_path, compress=True)
```

### Uploading the model to Algorithmia
Now let's call the Algorithmia utility function to take our saved model from its local path and put it on a data container on Algorithmia. As you'll remember, our algorithm script will be looking for the model to load at this data path.

We will call this function once we're happy with our model, that we'll develop soon.


```python
algorithmia_data_path = "data://asli/xgboost_demo"
algo_utility.upload_model_to_algorithmia(local_path, algorithmia_data_path, model_name)
```

### Time to test end to end!
Now we are up and ready and we have a perfectly scalable algorithm on Algorithmia, waiting for its visitors! Let's test it with one positive and one negative text and see how well it does. 
To send the request to our algorithm, we will use the algorithm calling function defined in the Algorithmia utility script, and we'll give it a string input.


```python
pos_test_input = "It doesn't work quite as expected. Not worth your money!"
#sentiment = get_review_sentiment(pos_test_input)
algo_result = algo_utility.call_latest_algo_version(pos_test_input)
print(algo_result.metadata)
print("Sentiment for the given text is: {}".format(algo_result.result["sentiment"]))
```

    Metadata(content_type='json',duration=0.020263526,stdout=None)
    Sentiment for the given text is: 0



```python
neg_test_input = "I am glad that I bought this. It works great!"
algo_result = algo_utility.call_latest_algo_version(neg_test_input)
print(algo_result.metadata)
print("Sentiment for the given text is: {}".format(algo_result.result["sentiment"]))
```

    Metadata(content_type='json',duration=0.018224132,stdout=None)
    Sentiment for the given text is: 1

