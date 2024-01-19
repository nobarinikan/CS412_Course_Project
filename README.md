# CS412 Machine Learning Course Project
***
## Project Description

The project aims to predict homework scores using various machine learning models. The approach involves extracting and preprocessing data from conversations, employing feature engineering techniques to enrich the dataset, and then applying multiple regression models to predict the scores. This multidimensional analysis ensures a thorough understanding and effective prediction of homework scores based on the given data.

The work presented is for the CS412 Machine Learning course at Sabanci University.

## Table of Contents
1. [General Info](#general-info)
2. [Methodology](#methodology)
3. [Results](#results)
4. [Team Contributions](#teamcontributions)
### General Info
***
## Methodology
***
A list of methodologies used within the project:

* Data Preparation

The data extracted from HTML files that store the ChatGPT conversations of students. The HTML files are read, parsed with BeautifulSoup and conversations are extracted based on specific patterns. THe code is extracted from file path, acting as a key in the “code2convos” dictionary where results are stored. If there are errors, detailed information is recorded in the log file, aiding in debugging and analysis. Duplicated keys are removed to prevent overwriting. 

Texts in the dictionary are preprocessed by several steps, including lowercasing, removal of punctuation and special characters, stopword removal, stemming, and lemmatization. Tokenized words are then rejoined into a single string. The dictionary is updated with the preprocessed version.

Prompts written by students are matched with questions given in the assignment by the simple term frequency vectorizing method. The distances between vectors representing prompts and questions are calculated. Mapping between file codes and their cosine similarity scores for each question is done.

Since we ran multiple training models for the next parts, tokenization and vectorization of data were needed for the ones which require a numerical format as it is a common preprocessing step for natural language processing tasks. 
  
```ruby
print(snippet here)
```

* Feature Engineering (Links to Code Snippets?)
  
* Select Features (Links to Code Snippets?)
  
* Different Models trainings (NN, randomForest etc.) (Links to Code Snippets?)
## Results
***
Discussion and conclusion

## Team Contributions
***
Give instructions on how to collaborate with your project.
> Maybe you want to write a quote in this part. 
> Should it encompass several lines?
> This is how you do it.


### Screenshot
<p align="center">
  <img src="Plots/image.png" alt="Ornek" width="50%">
</p>
