# Getting started with Xanthus

## What is Xanthus?

Xanthus is a Neural Recommender package written in Python. It started life as a personal project to take an academic ML paper and translate it into a 'production-ready' software package and to replicate the results of the paper along the way. It uses Tensorflow 2.0 under the hood, and makes extensive use of the Keras API. If you're interested, the original authors of [the paper that inspired this project](https://dl.acm.org/doi/10.1145/3038912.3052569) provided code for their experiments, and this proved valuable when starting this project. 

However, while it is great that they provided their code, the repository isn't maintained, the code uses an old versions of Keras (and Theano!), it can be a little hard for beginners to get to grips with, and it's very much tailored to produce the results in their paper. All fair enough, they wrote a great paper and published their workings. Admirable stuff. Xanthus aims to make it super easy to get started with the work of building a neural recommendation system, and to scale the techniques in the original paper (hopefully) gracefully with you as the complexity of your applications increase.

This notebook will walk you through a basic example of using Xanthus to predict previously unseen movies to a set of users using the classic 'Movielens' recommender dataset. The [original paper](https://dl.acm.org/doi/10.1145/3038912.3052569) tests the architectures in this paper as part of an _implicit_ recommendation problem. You'll find out more about what this means later in the notebook. In the meantime, it is worth remembering that the examples in this notebook make the same assumption.

Ready for some code?

## Loading a sample dataset

Ah, the beginning of a brand new ML problem. You'll need to download the dataset first. You can use the Xanthus `download.movielens` utility to download, unzip and save your Movielens data.


```python
from xanthus import datasets

datasets.movielens.download(version="ml-latest-small", output_dir="data")
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




    <xanthus.datasets.encoder.DatasetEncoder at 0x122228a20>



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

With your datasets ready, you can build and fit your model. In the example, the `GeneralizedMatrixFactorization` (or `GMFModel`) is used. If you're not sure what a GMF model is, be sure to check out the original paper, and the GMF class itself in the Xanthus docs. Anyway, here's how you set it up: 


```python
from xanthus.models import GeneralizedMatrixFactorization as GMFModel

model = GMFModel(train_ds.user_dim, train_ds.item_dim, factors=64)
model.compile(optimizer="adam", loss="binary_crossentropy")
```

So what's going on here? Well, `GMFModel` is a _subclass_ of the Keras `Model` class. Consequently, is shares the same interface. You will initialize your model with specific information (in this case information related to the size of the user and item input vectors and the size of the latent factors you're looking to compute), compile the model with a given loss and optimizer, and then train it. Straightforward enough, eh? In principle, you can use `GMFModel` however you'd use a 'normal' Keras model.

You're now ready to fit your model. You can do this with:


```python
# prepare training data
users_x, items_x, y = train_ds.to_components(
    negative_samples=4
)
model.fit([users_x, items_x], y, epochs=5)
```

    Epoch 1/5
    5729/5729 [==============================] - 7s 1ms/step - loss: 0.5001
    Epoch 2/5
    5729/5729 [==============================] - 7s 1ms/step - loss: 0.3685
    Epoch 3/5
    5729/5729 [==============================] - 7s 1ms/step - loss: 0.2969
    Epoch 4/5
    5729/5729 [==============================] - 7s 1ms/step - loss: 0.2246
    Epoch 5/5
    5729/5729 [==============================] - 7s 1ms/step - loss: 0.1581





    <tensorflow.python.keras.callbacks.History at 0x144717dd8>



Remember that (as with any ML model) you'll want to tweak your hyperparameters (e.g. `factors`, regularization, etc.) to optimize your model's performance on your given dataset. The example model here is just a quick un-tuned model to show you the ropes.

## Evaluating the model

Now to diagnose how well your model has done. The evaluation protocol here is set up in accordance with the methodology outlined in [the original paper](). To get yourself ready to generate some scores, you'll need to run:


```python
from xanthus.evaluate import create_rankings

users, items = create_rankings(
    test_ds, train_ds, output_dim=1, n_samples=100, unravel=True
)
```

So, what's going on here? First, you're importing the `create_rankings` function. This implements a sampling approach used be _He et al_ in their work. The idea is that you evaluate your model on the user-item pairs in your test set, and for each 'true' user-item pair, you sample `n_samples` negative instances for that user (i.e. items they haven't interacted with). In the case of the `create_rankings` function, this produces and array of shape `n_users, n_samples + 1`. Concretely, for each user, you'll get an array where the first element is a positive sample (something they _did_ interact with) and `n_samples` negative samples (things they _did not_ interact with). 

The rationale here is that by having the model rank these `n_samples + 1` items for each user, you'll be able to determine whether your model is learning an effective ranking function -- the positive sample _should_ appear higher in the recommendations than the negative results if the model is doing it's job. Here's how you can rank these sampled items:


```python
from xanthus.models import utils
test_users, test_items, _ = test_ds.to_components(shuffle=False)

scores = model.predict([users, items], verbose=1, batch_size=256)
recommended = utils.reshape_recommended(users.reshape(-1, 1), items.reshape(-1, 1), scores, 10, mode="array")
```

    240/240 [==============================] - 0s 540us/step


And finally for the evaluation, you can use the `score` function and the provided `metrics` in the Xanthus `evaluate` subpackage. Here's how you can use them:


```python
from xanthus.evaluate import score, metrics

print("t-nDCG", score(metrics.truncated_ndcg, test_items, recommended).mean())
print("HR@k", score(metrics.precision_at_k, test_items, recommended).mean())
```

    t-nDCG 0.4719391834962755
    HR@k 0.7351973684210527


Looking okay. Good work. Going into detail on how the metrics presented here work is beyond the scope of this notebook. If you're interested in what is going on here, make sure to check out the docs (docstrings) in the Xanthus package itself.

## The fun bit

After all of that, it is time to see what you've won. Exciting times. You can generate recommendations for your users _from unseen items_ by using the following:


```python
scores = model.predict([users, items], verbose=1, batch_size=256)
recommended = utils.reshape_recommended(users.reshape(-1, 1), items.reshape(-1, 1), scores, 10, mode="array")
```

    240/240 [==============================] - 0s 578us/step


Recall that the first 'column' in the `items` array corresponds to positive the positive sample for a user. You can skip that here. So now you have a great big array of integers. Not as exciting as you'd hoped? Fair enough. Xanthus provides a utility to convert the outputs of your model predictions into a more readable Pandas `DataFrame`. Specifically, your `DatasetEncoder` has the handy `to_df` method for just this job. Give it a set of _encoded_ users and a list of _encoded_ items for each user, and it'll build you a nice `DataFrame`. Here's how:


```python
recommended_df = encoder.to_df(test_users.flatten(), recommended)
recommended_df.head(25)
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
      <th>id</th>
      <th>item_0</th>
      <th>item_1</th>
      <th>item_2</th>
      <th>item_3</th>
      <th>item_4</th>
      <th>item_5</th>
      <th>item_6</th>
      <th>item_7</th>
      <th>item_8</th>
      <th>item_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Saint, The (1997)</td>
      <td>Fisher King, The (1991)</td>
      <td>Lost Boys, The (1987)</td>
      <td>West Side Story (1961)</td>
      <td>Harry Potter and the Sorcerer's Stone (a.k.a. ...</td>
      <td>Courage Under Fire (1996)</td>
      <td>Thing, The (1982)</td>
      <td>Tin Cup (1996)</td>
      <td>Mask of Zorro, The (1998)</td>
      <td>On Her Majesty's Secret Service (1969)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Seven (a.k.a. Se7en) (1995)</td>
      <td>Django Unchained (2012)</td>
      <td>Kung Fury (2015)</td>
      <td>There's Something About Mary (1998)</td>
      <td>Hanna (2011)</td>
      <td>Crash (2004)</td>
      <td>The Boss Baby (2017)</td>
      <td>Unbreakable (2000)</td>
      <td>Finding Dory (2016)</td>
      <td>Dr. Horrible's Sing-Along Blog (2008)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Four Weddings and a Funeral (1994)</td>
      <td>African Queen, The (1951)</td>
      <td>Jeremiah Johnson (1972)</td>
      <td>Fantastic Voyage (1966)</td>
      <td>Cobra (1986)</td>
      <td>Notorious (1946)</td>
      <td>Monsoon Wedding (2001)</td>
      <td>Heartbreak Ridge (1986)</td>
      <td>Miracle on 34th Street (1947)</td>
      <td>Raise the Titanic (1980)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Killing Fields, The (1984)</td>
      <td>Dracula (Bram Stoker's Dracula) (1992)</td>
      <td>Midnight Cowboy (1969)</td>
      <td>Pi (1998)</td>
      <td>Truman Show, The (1998)</td>
      <td>Out of Sight (1998)</td>
      <td>Last Emperor, The (1987)</td>
      <td>Once Were Warriors (1994)</td>
      <td>Everyone Says I Love You (1996)</td>
      <td>Nightmare Before Christmas, The (1993)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Harry Potter and the Sorcerer's Stone (a.k.a. ...</td>
      <td>This Is Spinal Tap (1984)</td>
      <td>Mask, The (1994)</td>
      <td>Airplane! (1980)</td>
      <td>Friday (1995)</td>
      <td>Star Trek IV: The Voyage Home (1986)</td>
      <td>Kelly's Heroes (1970)</td>
      <td>Inception (2010)</td>
      <td>What Dreams May Come (1998)</td>
      <td>For Love of the Game (1999)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Home Alone (1990)</td>
      <td>Supercop (Police Story 3: Supercop) (Jing cha ...</td>
      <td>Dragonheart (1996)</td>
      <td>Funny People (2009)</td>
      <td>Kazaam (1996)</td>
      <td>Red Dawn (1984)</td>
      <td>Patch Adams (1998)</td>
      <td>Ruthless People (1986)</td>
      <td>Footloose (1984)</td>
      <td>Sleepers (1996)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>There's Something About Mary (1998)</td>
      <td>Gladiator (2000)</td>
      <td>Ferris Bueller's Day Off (1986)</td>
      <td>Crimson Tide (1995)</td>
      <td>Die Hard: With a Vengeance (1995)</td>
      <td>Shakespeare in Love (1998)</td>
      <td>Young Frankenstein (1974)</td>
      <td>Batman (1989)</td>
      <td>Game, The (1997)</td>
      <td>Christmas Story, A (1983)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Firm, The (1993)</td>
      <td>Dangerous Minds (1995)</td>
      <td>Few Good Men, A (1992)</td>
      <td>Who Framed Roger Rabbit? (1988)</td>
      <td>Bonnie and Clyde (1967)</td>
      <td>Superman (1978)</td>
      <td>Carlito's Way (1993)</td>
      <td>Rocky III (1982)</td>
      <td>Trainspotting (1996)</td>
      <td>What's Love Got to Do with It? (1993)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Cinema Paradiso (Nuovo cinema Paradiso) (1989)</td>
      <td>Lord of the Rings: The Fellowship of the Ring,...</td>
      <td>Finding Nemo (2003)</td>
      <td>Hangover, The (2009)</td>
      <td>Running Man, The (1987)</td>
      <td>Ben-Hur (1959)</td>
      <td>Talented Mr. Ripley, The (1999)</td>
      <td>Sliding Doors (1998)</td>
      <td>Kagemusha (1980)</td>
      <td>Some Like It Hot (1959)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>Young Frankenstein (1974)</td>
      <td>Batman &amp; Robin (1997)</td>
      <td>Tangled (2010)</td>
      <td>Louis C.K.: Hilarious (2010)</td>
      <td>Pacific Rim (2013)</td>
      <td>Planet of the Apes (2001)</td>
      <td>American Pie (1999)</td>
      <td>Guardians of the Galaxy (2014)</td>
      <td>X-Men (2000)</td>
      <td>28 Days Later (2002)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>Heat (1995)</td>
      <td>Analyze This (1999)</td>
      <td>To Wong Foo, Thanks for Everything! Julie Newm...</td>
      <td>Mystery, Alaska (1999)</td>
      <td>I, Robot (2004)</td>
      <td>Invincible (2006)</td>
      <td>Gandhi (1982)</td>
      <td>Galaxy Quest (1999)</td>
      <td>Training Day (2001)</td>
      <td>Romy and Michele's High School Reunion (1997)</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>'burbs, The (1989)</td>
      <td>Hercules (1997)</td>
      <td>Payback (1999)</td>
      <td>Three Men and a Baby (1987)</td>
      <td>Enemy of the State (1998)</td>
      <td>Top Gun (1986)</td>
      <td>White Squall (1996)</td>
      <td>Dumbo (1941)</td>
      <td>Amistad (1997)</td>
      <td>Quiz Show (1994)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>Die Hard (1988)</td>
      <td>Outbreak (1995)</td>
      <td>Zootopia (2016)</td>
      <td>I Am Legend (2007)</td>
      <td>Kate &amp; Leopold (2001)</td>
      <td>Lost in Translation (2003)</td>
      <td>Battle Royale (Batoru rowaiaru) (2000)</td>
      <td>Mallrats (1995)</td>
      <td>Gangs of New York (2002)</td>
      <td>Lethal Weapon (1987)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>Don Juan DeMarco (1995)</td>
      <td>Jungle Book, The (1994)</td>
      <td>River Wild, The (1994)</td>
      <td>National Treasure (2004)</td>
      <td>Super Troopers (2001)</td>
      <td>Amistad (1997)</td>
      <td>Django Unchained (2012)</td>
      <td>Son in Law (1993)</td>
      <td>Chocolat (1988)</td>
      <td>Doctor Zhivago (1965)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>Raiders of the Lost Ark (Indiana Jones and the...</td>
      <td>Silence of the Lambs, The (1991)</td>
      <td>Hobbit: The Desolation of Smaug, The (2013)</td>
      <td>50 First Dates (2004)</td>
      <td>Prometheus (2012)</td>
      <td>Vertigo (1958)</td>
      <td>Fear and Loathing in Las Vegas (1998)</td>
      <td>Billy Madison (1995)</td>
      <td>Knocked Up (2007)</td>
      <td>Eraser (1996)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>Life Is Beautiful (La Vita è bella) (1997)</td>
      <td>Ed Wood (1994)</td>
      <td>Wallace &amp; Gromit: A Close Shave (1995)</td>
      <td>Pinocchio (1940)</td>
      <td>Fast Times at Ridgemont High (1982)</td>
      <td>Corpse Bride (2005)</td>
      <td>Basic Instinct (1992)</td>
      <td>Billy Elliot (2000)</td>
      <td>Before Sunrise (1995)</td>
      <td>Gia (1998)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>Dr. Strangelove or: How I Learned to Stop Worr...</td>
      <td>RoboCop (1987)</td>
      <td>The Devil's Advocate (1997)</td>
      <td>eXistenZ (1999)</td>
      <td>Robin Hood: Men in Tights (1993)</td>
      <td>Bourne Supremacy, The (2004)</td>
      <td>Blair Witch Project, The (1999)</td>
      <td>Sicario (2015)</td>
      <td>The Count of Monte Cristo (2002)</td>
      <td>Indiana Jones and the Kingdom of the Crystal S...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>Run Lola Run (Lola rennt) (1998)</td>
      <td>WALL·E (2008)</td>
      <td>Fury (2014)</td>
      <td>Fish Called Wanda, A (1988)</td>
      <td>Peter Pan (1953)</td>
      <td>Braveheart (1995)</td>
      <td>Scanner Darkly, A (2006)</td>
      <td>The Hobbit: The Battle of the Five Armies (2014)</td>
      <td>Day After Tomorrow, The (2004)</td>
      <td>Cowboy Bebop: The Movie (Cowboy Bebop: Tengoku...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>Like Water for Chocolate (Como agua para choco...</td>
      <td>Crocodile Dundee (1986)</td>
      <td>Speed (1994)</td>
      <td>Congo (1995)</td>
      <td>Ruthless People (1986)</td>
      <td>101 Dalmatians (One Hundred and One Dalmatians...</td>
      <td>Pleasantville (1998)</td>
      <td>Killing Fields, The (1984)</td>
      <td>Shanghai Noon (2000)</td>
      <td>*batteries not included (1987)</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>Dodgeball: A True Underdog Story (2004)</td>
      <td>Harry Potter and the Goblet of Fire (2005)</td>
      <td>Citizen Kane (1941)</td>
      <td>13 Going on 30 (2004)</td>
      <td>Star Wars: Episode III - Revenge of the Sith (...</td>
      <td>Alien: Resurrection (1997)</td>
      <td>Lord of the Rings, The (1978)</td>
      <td>Mask, The (1994)</td>
      <td>20,000 Leagues Under the Sea (1954)</td>
      <td>Over the Hedge (2006)</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>Gone Girl (2014)</td>
      <td>Whiplash (2014)</td>
      <td>Grown Ups 2 (2013)</td>
      <td>Mercury Rising (1998)</td>
      <td>(500) Days of Summer (2009)</td>
      <td>Apocalypto (2006)</td>
      <td>Red Riding Hood (2011)</td>
      <td>Creepshow (1982)</td>
      <td>World Is Not Enough, The (1999)</td>
      <td>Dear Zachary: A Letter to a Son About His Fath...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>Bowling for Columbine (2002)</td>
      <td>Memento (2000)</td>
      <td>Blow (2001)</td>
      <td>Pianist, The (2002)</td>
      <td>Star Wars: Episode II - Attack of the Clones (...</td>
      <td>Apollo 13 (1995)</td>
      <td>Rudy (1993)</td>
      <td>Romeo and Juliet (1968)</td>
      <td>Beetlejuice (1988)</td>
      <td>X-Men: Days of Future Past (2014)</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>Spirited Away (Sen to Chihiro no kamikakushi) ...</td>
      <td>Excalibur (1981)</td>
      <td>Once Upon a Time in America (1984)</td>
      <td>Austin Powers: The Spy Who Shagged Me (1999)</td>
      <td>Jerk, The (1979)</td>
      <td>Cat on a Hot Tin Roof (1958)</td>
      <td>Wallace &amp; Gromit: The Wrong Trousers (1993)</td>
      <td>You Can Count on Me (2000)</td>
      <td>Out of Sight (1998)</td>
      <td>Harry Potter and the Deathly Hallows: Part 1 (...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>Léon: The Professional (a.k.a. The Professiona...</td>
      <td>King Kong (2005)</td>
      <td>Lethal Weapon 3 (1992)</td>
      <td>Star Trek Beyond (2016)</td>
      <td>Star Trek: Nemesis (2002)</td>
      <td>Road to Perdition (2002)</td>
      <td>To Kill a Mockingbird (1962)</td>
      <td>A-Team, The (2010)</td>
      <td>Home (2015)</td>
      <td>Stripes (1981)</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>Shawshank Redemption, The (1994)</td>
      <td>WALL·E (2008)</td>
      <td>Walk the Line (2005)</td>
      <td>Town, The (2010)</td>
      <td>Cars (2006)</td>
      <td>Elite Squad: The Enemy Within (Tropa de Elite ...</td>
      <td>Mulholland Falls (1996)</td>
      <td>Motorcycle Diaries, The (Diarios de motociclet...</td>
      <td>Rainmaker, The (1997)</td>
      <td>Dallas Buyers Club (2013)</td>
    </tr>
  </tbody>
</table>
</div>



## That's a wrap

And that's it for this example. Be sure to raise any issues you have [on GitHub](https://github.com/markdouthwaite/xanthus), or get in touch [on Twitter](https://twitter.com/MarklDouthwaite).
