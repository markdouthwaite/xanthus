# Getting started with Xanthus

## What is Xanthus?

Xanthus is a Neural Recommender package written in Python. It started life as a personal project to take an academic ML paper and translate it into a 'production-ready' software package and to replicate the results of the paper along the way. It uses Tensorflow 2.0 under the hood, and makes extensive use of the Keras API. If you're interested, the original authors of [the paper that inspired this project]() provided code for their experiments, and this proved valuable when starting this project. 

However, while it is great that they provided their code, the repository isn't maintained, the code uses an old versions of Keras (and Theano!), it can be a little hard for beginners to get to grips with, and it's very much tailored to produce the results in their paper. All fair enough, they wrote a great paper and published their workings. Admirable stuff. Xanthus aims to make it super easy to get started with the work of building a neural recommenation system, and to scale the techniques in the original paper (hopefully) gracefully with you as the complexity of your applications increase.

This notebook will walk you through a basic example of using Xanthus to predict previously unseen movies to a set of users using the classic 'Movielens' recommender dataset. The [original paper]() tests the architectures in this paper as part of an _implicit_ recommendation problem. You'll find out more about what this means later in the notebook. In the meantime, it is worth remembering that the examples in this notebook make the same assumption.

Ready for some code?

## Loading a sample dataset

Ah, the beginning of a brand new ML problem. You'll need to download the dataset first. You can use the Xanthus `download.movielens` utility to download, unzip and save your Movielens data.


```python
from xanthus.datasets import download

download.movielens(version="latest-small", output_dir="data")
```

Time to crack out Pandas and load some CSVs. You know the drill. 


```python
import pandas as pd

ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
movies = pd.read_csv("data/ml-latest-small/movies.csv")
```

Let's take a look at the data we've loaded. Here's the movies dataset:


```python
movies.head()
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>



As you can see, you've got the unique identifier for your movies, the title of the movie in human-readable format, and then the column `genres` that has a string containing a set of associated genres for the given movie. Straightforward enough. And hey, that `genres` column might come in handy at some point...

On to the `ratings` frame. Here's what is in there:


```python
ratings.head()
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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>964982703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>4.0</td>
      <td>964981247</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>6</td>
      <td>4.0</td>
      <td>964982224</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>5.0</td>
      <td>964983815</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>5.0</td>
      <td>964982931</td>
    </tr>
  </tbody>
</table>
</div>



First up, you've got a `userId` corresponding to the unique user identifier, and you've got the `movieId` corresponding to the unique movie identifier (this maps onto the `movieId` column in the `movies` frame, above). You've also got a `rating` field. This is associated with the user-assigned rating for that movie. Finally, you have the `timestamp` -- the date at which the user rated the movie. For future reference, you can convert from this timestamp to a 'human readable' date with:


```python
from datetime import datetime

datetime.fromtimestamp(ratings.iloc[0]["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
```




    '2000-07-30 19:45:03'



Thats your freebie for the day. Onto getting the data ready for training your recommender model.

## Data preparation

Xanthus provides a few utilities for getting your recommender up and running. One of the more ubiquitous utilities is the `Dataset` class, and its related `DatasetEncoder` class. At the time of writing, the `Dataset` class assumes your 'ratings' data is in the format `user`, `item`, `rating`. You can rename the sample data to be in this format with:


```python
ratings = ratings.rename(columns={"userId": "user", "movieId": "item"})
```

Next, you might find it helpful to re-map the movie IDs (now under the `item` column) to be the `titles` in the `movies` frame. This'll make it easier for you to see what the recommender is recommending! Don't do this for big datasets though -- it can get very expensive very quickly! Anyway, remap the `item` column with:


```python
title_mapping = dict(zip(movies["movieId"], movies["title"]))
ratings.loc[:, "item"] = ratings["item"].apply(lambda _: title_mapping[_])
ratings.head(2)
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
      <th>user</th>
      <th>item</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>4.0</td>
      <td>964982703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Grumpier Old Men (1995)</td>
      <td>4.0</td>
      <td>964981247</td>
    </tr>
  </tbody>
</table>
</div>



A little more meaningful, eh? For this example, you are going to be looking at _implicit_ recommendations, so should also remove clearly negative rating pairs from the dataset. You can do this with:


```python
ratings = ratings[ratings["rating"] > 3.0]
```

### Leave one out protocol

As with any ML model, it is important to keep a held-out sample of your dataset to evaluate your model's performance. This is naturally important for recommenders too. However, recommenders differ slightly in that we are often interested in the recommender's ability to _rank_ candidate items in order to surface the most relevant content to a user. Ultimately, the essence of recommendation problems is search, and getting relevant items in the top `n` search results is generally the name of the game -- absolute accuracy can often be a secondary consideration.

One common way of evaluating the performance of a recommender model is therefore to create a test set by sampling `n` items from each user's `m` interactions (e.g. movie ratings), keeping `m-n` interactions in the training set and putting the 'left out' `n` samples in the test set. The thought process then goes that when evaluating a model on this test set, you should see the model rank the 'held' out samples more highly in the results (i.e. it has started to learn a user's preferences). 

The 'leave one out' protocol is a specific case of this approach where `n=1`. Concretely, when creating a test set using 'leave one out', you withold a single interaction from each user and put these in your test set. You then place all other interactions in your training set. To get you going, Xanthus provides a utility function called -- funnily enough -- `leave_one_out` under the `evaluate` subpackage. You can import it and use it as follows:


```python
from xanthus.evaluate import leave_one_out

train_df, test_df = leave_one_out(ratings, shuffle=True, deduplicate=True)
```

You'll notice that there's a couple of things going on here. Firstly, the function returns the input interactions frame (in this case `ratings`) and splits it into the two datasets as expected. Fair enough. We then have two keyword arguments `shuffle` and `deduplicate`. The argument `shuffle` will -- you guessed it -- shuffle your dataset before sampling interactions for your test set. This is set to `True` by default, so it is shown here for the purpose of being explicit. The second argument is `deduplicate`. This does what you might expect too -- it strips any cases where a user interacts with a specific item more than once (i.e. a given user-item pair appears more than once).

As discussed above, the `leave_one_out` function is really a specific version of a more general 'leave `n` out' approach to splitting a dataset. There's also other ways you might want to split datasets for recommendation problems. For many of those circumstances, Xanthus provides a more generic `split` function. This was inspired by Azure's [_Recommender Split_](https://docs.microsoft.com/en-us/azure/machine-learning/studio-module-reference/split-data-using-recommender-split#:~:text=The%20Recommender%20Split%20option%20is,user%2Ditem%2Drating%20triples) method in Azure ML Studio. There are a few important tweaks in the Xanthus implementation, so make sure to check out that functions documentation if you're interested.

Anyway, time to build some datasets.

## Introducing the `Dataset`

Like other ML problems, recommendation problems typically need to create encoded representations of a domain in order to be passed into a model for training and evaluation. However, there's a few aspects of recommendation problems that can make this problem particularly fiddly. To help you on your way, Xanthus provides a few utilities, including the `Dataset` class and the `DatasetEncoder` class. These structures are designed to take care of the fiddliness for you. They'll build your input vectors (including with metadata, if you provide it -- more on that later) and sparse matrices as required. You shouldn't need to touch a thing. 

Here's how it works. First, your 'train' and 'test' datasets are going to need to share the same encodings, right? Otherwise they'll disagree on whether `Batman Forever (1995)` shares the same encoding across the datasets, and that would be a terrible shame. To create your `DatasetEncoder` you can do this:


```python
from xanthus.datasets import DatasetEncoder

encoder = DatasetEncoder()
encoder.fit(ratings["user"], ratings["item"])
```




    <xanthus.datasets.encoder.DatasetEncoder at 0x128778da0>



This encoder will store all of the unique encodings of every user and item in the `ratings` set. Notice that you're passing in the `ratings` set here, as opposed to either train or test. This makes doubly sure you're creating encodings for every user-item pair in the dataset. To check this has worked, you can call the `transform` method on the encoder like this:


```python
encoder.transform(items=["Batman Forever (1995)"])
```




    {'items': array([1694], dtype=int32)}



The naming conventions on the `DatasetEncoder` are deliberately reminicent of the methods on Scikit-Learn encoders, just to help you along with using them. Now you've got your encoder, you can create your `Dataset` objects:


```python
from xanthus.datasets import Dataset, utils

train_ds = Dataset.from_df(train_df, normalize=utils.as_implicit, encoder=encoder)
test_ds = Dataset.from_df(test_df, normalize=utils.as_implicit, encoder=encoder)
```

Let's unpack what's going on here. The `Dataset` class provides the `from_df` class method for quickly constructing a `Dataset` from a 'raw' Pandas `DataFrame`. You want to create a train and test dataset, hence creating two separate `Dataset` objects using this method. Next, you can see that the `encoder` keyword argument is passed in to the `from_df` method. This ensures that each `Dataset` maintains a reference to the _same_ `DatasetEncoder` to ensure consistency when used. The final argument here is `normalize`. This expects a callable object (e.g. a function) that scales the `rating` column (if provided). In the case of this example, the normalization is simply to treat the ratings as an implicit recommendation problem (i.e. all zero or one). The `utils.as_implicit` function simply sets all ratings to one. Simple enough, eh?

And that is it for preparing your datasets for modelling, at least for now. Time for some Neural Networks.

## Getting neural

With your datasets ready, you can build and fit your model. In the example, the `GeneralizedMatrixFactorizationModel` (or `GMFModel`) is used. If you're not sure what a GMF model is, be sure to check out the original paper, and the GMF class itself in the Xanthus docs. Anyway, here's how you set it up: 


```python
from xanthus.models import GeneralizedMatrixFactorizationModel as GMFModel

fit_params = dict(epochs=10, batch_size=256)

model = GMFModel(
    fit_params=fit_params, n_factors=32, negative_samples=4
)
```

What is going on here, you ask? Good question. First, you import the `GeneralizedMatrixFactorizationModel` as any other object. You then define `fit_params` -- fit parameters -- to define the training loop used by the Keras optimizer. All Xanthus neural recommender models inherit from the base `NeuralRecommenderModel` class. By default, this class (and therefore all child classes) utilize the `Adam` optimizer. You can configure this to use any optimizer you wish though!

After the `fit_param`, the `GeneralizedMatrixFactorizationModel` is initialized. There are two further keyword arguments here, `n_factors` and `negative_samples`. In the former case, `n_factors` refers to the size of the latent factor space encoded by the model. The larger the number, the more expressive the model -- to a point. In the latter case, `negative_samples` configures the sampling pointwise sampling policy outlined by [He et al](). In practice, the model will be trained by sampling 'negative' instances for each positive instance in the set. In other words: for each user-item pair with a positive rating (in this case one -- remember `utils.as_implicit`?), a given number of `negative_samples` will be drawn that the user _did not_ interact with. This is resampled in each epoch. This helps the model learn more general patterns, and to avoid overfitting. Empirically, it makes quite a difference over other sampling approaches. If you're interested, you should look at the [pairwise loss used in Bayesian Personalized Ranking (BPR)]().

You're now ready to fit your model. You can do this with:


```python
model.fit(train_ds)
```

    1075/1075 [==============================] - 2s 2ms/step - loss: 0.4895 - val_loss: 0.3513
    Epoch 2/2
    1075/1075 [==============================] - 2s 2ms/step - loss: 0.3343 - val_loss: 0.3284
    Epoch 3/3
    1075/1075 [==============================] - 2s 2ms/step - loss: 0.3087 - val_loss: 0.3057
    Epoch 4/4
     565/1075 [==============>...............] - ETA: 0s - loss: 0.2908

Remember that (as with any ML model) you'll want to tweak your hyperparameters (e.g. `n_factor`, regularization, etc.) to optimize your model's performance on your given dataset. The example model here is just a quick un-tuned model to show you the ropes.

## Evaluating the model

Now to diagnose how well your model has done. The evaluation protocol here is set up in accordance with the methodology outlined in [the original paper](). To get yourself ready to generate some scores, you'll need to run:


```python
from xanthus.evaluate import he_sampling

_, test_items, _ = test_ds.to_components(shuffle=False)
users, items = he_sampling(test_ds, train_ds, n_samples=200)
```

So, what's going on here? First, you're importing the `he_sampling` function. This implements a sampling approach used be [He et al.]() in their work. The idea is that you evaluate your model on the user-item pairs in your test set, and for each 'true' user-item pair, you sample `n_samples` negative instances for that user (i.e. items they haven't interacted with). In the case of the `he_sampling` function, this produces and array of shape `n_users, n_samples + 1`. Concretely, for each user, you'll get an array where the first element is a positive sample (something they _did_ interact with) and `n_samples` negative samples (things they _did not_ interact with). 

The rationale here is that by having the model rank these `n_samples + 1` items for each user, you'll be able to determine whether your model is learning an effective ranking function -- the positive sample _should_ appear higher in the recommendations than the negative results if the model is doing it's job. Here's how you can rank these sampled items:


```python
recommended = model.predict(test_ds, users=users, items=items, n=10)
```

And finally for the evaluation, you can use the `score` function and the provided `metrics` in the Xanthus `evaluate` subpackage. Here's how you can use them:


```python
from xanthus.evaluate import score, metrics

print("t-nDCG", score(metrics.truncated_ndcg, test_items, recommended).mean())
print("HR@k", score(metrics.precision_at_k, test_items, recommended).mean())
```

Looking okay. Good work. Going into detail on how the metrics presented here work is beyond the scope of this notebook. If you're interested in what is going on here, make sure to check out the docs (docstrings) in the Xanthus package itself.

## The fun bit

After all of that, it is time to see what you've won. Exciting times. You can generate recommendations for your users _from unseen items_ by using the following:


```python
recommended = model.predict(users=users, items=items[:, 1:], n=5)
```

Recall that the first 'column' in the `items` array corresponds to positive the positive sample for a user. You can skip that here. So now you have a great big array of integers. Not as exciting as you'd hoped? Fair enough. Xanthus provides a utility to convert the outputs of your model predictions into a more readable Pandas `DataFrame`. Specifically, your `DatasetEncoder` has the handy `to_df` method for just this job. Give it a set of _encoded_ users and a list of _encoded_ items for each user, and it'll build you a nice `DataFrame`. Here's how:


```python
recommended_df = encoder.to_df(users, recommended)
recommended_df.head(25)
```

## That's a wrap

And that's it for this example. Be sure to raise any issues you have [on GitHub](https://github.com/markdouthwaite/xanthus), or get in touch [on Twitter](https://twitter.com/MarklDouthwaite).
