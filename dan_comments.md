Code Review Comments

There's a lot of comments, but don't take it personally :).

I'm reviewing this as if it's intended to be an open-source package, but of course it's a draft version of ideas right now. The comments are written with more of a vision of how the project should look at the end if you wish to have it open-sourced.

The code looks really nice already, and I think with some small changes we can get some really good habits that will benefit you enormously not just here but also in your future career. 


## Structure

Having numbers in the folder names is something I wouldn't use, as it's a little ugly to read. As a package, all files will have essentially equal weight and there won't be an order in which you need to look at folders.

It's recommended to have a `src/` folder in which you place your functions. 

For example, the tree of your project will eventually look something like this:

├── README.md

├── docs/  # Model Documentation

├── notebooks

│   └── demo  # Public Notebooks for use by all, including model generation, feature exploration etc.

├── setup.py  # For using this repo as a package

├── src  # All main code is in here

│   ├── features  # Creating all features

│   └── utils  # Helpful functions

└── tests/  # All tests for functions in src/


See projects like [TACO](https://dev.azure.com/IngEurCDaaS01/IngOne/_git/P01908-taco) as a guide

This means that you can just do:
``````
from JanProject import wavelet_function
``````
And then here you could have e.g.:
src/signal_processing.py 
src/wavelet.py
etc.

Notice that tests/ is a folder above - it's a very good idea to write unit tests for each function you've created to ensure it's doing what it should be doing. Take a look at `pytest`

## Data

It's not recommended to have a large dataset on git, especially for an open source package, as you don't want every user to download huge files every time they use pip install. You can instead put in the Readme the dataset that was used, and the steps that you took to download it. If you do want to include some data, which we do in skorecard, then it's good to take a small sample of it

## gitignore

Great that you're already familiar with this!

## Main.py

- Minor detail, usually it's a small 'm', main.py
- It's good to group imports together from the same package and absolutely remove imports that are not used
- parameter names should be explicit, and it doesn't matter if they get quite long. For example, when I see the parameter `fillBalMeth`I think of filling balls with meth :D, but I'm not sure what it actually means.
- The same applies for function names - it should be clear what they're doing. e.g. the function 'monthly' doesn't tell me what its purpose is
- It's good to have 1 function doing 1 thing. I don't think the function `computeFeat` is needed, as you essentially just have a function in a function.
- This means that some functions, like `monthly` and `fillBalTran` (better name needed :)) would benefit from becoming more modularised.
- Also, make sure that you catch instances that can go wrong. e.g. in `computeFeat`, what happens if I put 'featType'='ice cream'?
- When certain errors are caught, it's good practice to, e.g. `raise ValueError('variable must be a string')` instead of using `print`
- Try to remove unnecessary params where possible. For example, your function:

``````
def computeBasicFeat(data):  
    "compute basic features of input data: min max avg skw krt std"
    fmin = data.min()
    fmax = data.max()
    favg = data.mean()
    fskw = data.skew()
    fkrt = data.kurt()
    fstd = data.std()
    # fsum = dataMonth.amount.sum()    
    features = [fmin, fmax, favg, fskw, fkrt, fstd]   
    return features
``````

could be rewritten:

``````
def computeBasicFeat(data):  
    "compute basic features of input data: min max avg skw krt std"  
    return [data.min(), data.max(), data.mean(), data.skew(), data.kurt(), data.std()] 
``````

- Regarding docstrings, we use [Google Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) as a neat, explicit way of telling the user what the functions do, with small extra comments above the lines if needed. It's currently difficult for me to understand what some of your functions are doing. For example, your function:

```````
def waveletDenoise(data, thresh = 0.63, wavelet='db2'):
    "removing noisy high frequencies from the input data by applying a wavelet threshold"
    thresh = thresh*np.nanmax(data.iloc[:,1].values)
    coeff = pywt.wavedec(data.iloc[:,1].values, wavelet, mode="periodization")
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
    data.iloc[:,1] = pywt.waverec(coeff, wavelet, mode="periodization")       
    return data
```````

Would benefit greatly from this structure:

``````
def waveletDenoise(data, thresh = 0.63, wavelet='db2'):
    """"
    This function removes noisy high frequencies from the input data by applying a wavelet threshold.
    
    Args:
        data: Must be of format blahblah
        threshold (float): The frequency threshold above which we remove all signal
        wavelet (str): I don't know what this is
        
    Returns:
    
        data: The original data with another column(?) which corresponds to...
    
    """
    # Calculate New threshold because blahblah
    thresh = thresh*np.nanmax(data.iloc[:,1].values)
    coeff = pywt.wavedec(data.iloc[:,1].values, wavelet, mode="periodization")
    
    # Explain in 1 line why this is needed:
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
    
    # Explain in 1 line why this is needed:
    data.iloc[:,1] = pywt.waverec(coeff, wavelet, mode="periodization")       
    return data
``````
- I see also in this function the option to change the mode. Currently it is set to 'periodization' - what if I want to use a different mode?
- By the way, for financial signals is there a sensible threshold we should pick here? Is it normalised or does it depend on the sizes of the transactions?
- As some of the main.py is not meant for open source, but instead benchmarking, I would make a separate python file here. Something like 'benchmarking.py' where you import your necessary functions and time and score them.

- Columns as function arguments: some of the functions assume that the data has a column 'amount', but this might be different for others using a different dataset. It's best to have the column as an argument, so:

``````
def computeBasic(data, column='amount'):
    featuresAmount = computeBasicFeat(data[column])
``````

It's not good practice to have code like this:
``````
def computeWavelet(data, depth):
    wavelet = pywt.wavedec(data, 'db2', level=int(depth)) <-this bit
``````
In this function, you're immediately reformatting the depth without telling the user. It's best to enforce the user to pick an integer

### Minimal Working Example

When it comes to others using your code, they won't be using the same dataset, so you want this to be flexible for other datasets. What would be great to have is a minimal working file, where you explicitly show:
- This is the format that your data must be in before you apply my functions
- This is how you easily apply my functions
- This is how you gauge the results of my functions

## Linting

To make your code even cleaner it's recommended to using a linting method, such as flake8 or black. I like black as it reformats the file for you:

``````
pip install black
black main.py
``````
This will make your code follow all of the recommended PEP8 guidelines.

## setup.py

Eventually the goal will be to have a setup.py such that the package is pip installable. Check out TACO as an example. In this small file, it contains all of the dependencies, contact details etc.

## README

This is used to convey to the user in a few sentences what the package is about, how to install it, and how to use it. You can also include the structure in it if you wish.
            
