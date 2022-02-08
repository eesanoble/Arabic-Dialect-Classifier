# Arabic-Dialect-Classifier

## Contents

1.  Technologies and Tools Used
2.  Introduction
3.  Use Cases
4.  Objectives
5.  Data Source
6.  Hypotheses
7.  Preparing the Arabic
8.  Our Datasets
9.  Arabic vs MSA models
10.  Dialect Classification Models
		* n-grams
		* Confusion Matrix
		*  Oversampling
		* Under the hood
16.  Conclusions
17.  Key Learnings
18.  Next Steps

## Technologies and Tools Used

Pandas, numpy, matplotlib, seaborn, camel_tools, scipy, RegEx, WordCloud, sklearn, scikitplot, arabic_reshaper, bidi, os, codecs, imblearn, XGBoost

## Introduction

Unlike the languages which are more commonly used in modern NLP, Arabic is *diglossic*. This means that it has two registers - a formal and an informal. The formal style of Arabic, known as MSA (Modern Standard Arabic) is the same across the Arab world. The informal style or dialect, which is what people actually speak in, is very geographically specific.

Until recently it was rare that you would see dialectical Arabic written down since a book, or newspaper article, or academic paper would almost always be written in MSA. However, thanks in part to the advent of social media, dialect is seen in its written form far more often. This provides translators (especially machine translators) with an issue. How do you know which 'Arabic' you are translating?


**![](https://lh6.googleusercontent.com/mjIQ7DNB_hjam3xSFItkAgdy3vZHE5E9A8kHf4nIi7SVmzbov_YYczQBokf-ZH1TzW-OnTQqF6ZgcLxO4XGAIukNrZ5Oo4Ow0_6_PKhU_hkxnB7_ZEK-bHCB-SkOzdwdYwByRjTzMDjs)**
## Use Cases

To be able to use the language of an Arabic Text itself to ascertain which dialect it is written would: 1- Aid translators, 2- Allow to geographically locate based on language, 3- Allow dialectologists some insight into what makes up a dialect.

## Objectives

The goal of this project is to create a classification model which uses the text of an Arabic Tweet to ascertain 1- if it is written in MSA or dialect and 2- which dialectical region a dialect tweet comes from. I would like to outperform Baseline Accuracy in both cases.

## Data Source

The data for this project was derived from a corpus of tweets put together for the purpose of the 2nd NADI (Nuanced Arabic Dialect Identification Shared Task).

The data was collected through the Twitter API using the location data to establish province and country of origin.

 (**See** *Abdul-Mageed, M., Zhang, C., Elmadany, A., Bouamor, H., & Habash, N. (2021). NADI 2021: The Second Nuanced Arabic Dialect Identification Shared Task. _ArXiv, abs/2103.08466_.*)

The Datasets I used were:

-   **Dataset 1:** Country-level **MSA** identification: A total of 21,000 tweets in MSA, covering 21 Arab countries.

-   **Dataset 2:** Country-level **Dialect** identification: A total of 21,000 tweets in dialect, covering 21 Arab countries.

## Hypotheses

My hypotheses are:

 1. It will be fairly easy to ascertain which tweets are in MSA and which are in dialect. This is because there are certain fairly common features of MSA which are very specific and aren't shared in any dialects.
 2. It will, however, be more difficult to differentiate between specific dialects because of shared features across dialects.
 3. In Bag of Words models certain common words will be used most in differentiating dialects. Words like 'فين', which means 'where' in Egyptian, and 'لماذا' which means 'why' in MSA will be obvious markers for a model.

## Preparing the Arabic

Arabic and NLP don't get along very well. This is for a number of reasons:

 1. **Script**
	 The script goes right to left and letters take different forms depending on where they are in a word.

 2. **Diacritics**
	 Vowels are optional in Arabic and are more often than not omitted.

 3. **Orthographic Ambiguity**
	 Because of the optional vowels, it is often difficult to tell what a word is without context.

 4. **Morphological Complexity**
	 Words look *very* different depending on how they are used grammatically. This makes stemming very difficult. (*Stemming is where you tell your model that although two words look different, you want to treat them as the same word - for example, ‘walk’ and ‘walks’.*)

 5. **Orthographic Inconsistency**
	 The same words can be spelt differently (like doughnut/donut).

There are a number of things we can do to solve some of these issues and make our models better:

 1. **Transliteration**
	This means mapping each Arabic letter onto a letter of the Roman alphabet.

 2. **Dediacritisation**
	This means removing all the vowels from words, so there is no variation in words that are the same because some are vowelled and some aren't.

 3. **Normalisation**
	Forcing all words to be spelt the same

 4. **Morphological Tokenisation**
 Splitting words up according to their morphological make up.

**![](https://lh6.googleusercontent.com/lFHV2O9I9bG82jNg9Bbt7Z6uk_EPEOi2J8zoU8hZ2Hjhm9pHih9BOzLRmUZSIdUmY4egKbx6IAJcw3ym0l1sKUKnXqC7ft38-lYqC4OTWIOf_lPbPTgsZxYcKbXbPowQJHhi5gpl9GCO)**


## Our Datasets

For our first model we want to differentiate between MSA and dialect. I labelled the datasets accordingly and had a look at the data.

Below are two WordClouds where the size of a word represents its frequency.

What they demonstrate is that although there are some differences between the two registers, the most common words are the same - they are after all the same language.

**![](https://lh5.googleusercontent.com/_-sx3lI_BmisubQtDv44zMWjOtMRiOU397CEsnr5r6LLf5JaC8uKoo6ZFmOGs9wvnQQRwMACX8pS9w0H3UIo8M_flTrbC8ukzBqoRs3RawmNrwP4IiBqS2c7RSMgcaEEVRl7CUI-n3_U)**

## Arabic vs MSA Models

I ran a few models with different types of tokenisations. The best model I produced was the Logistic Regression Model with Count Vectorisation (Accuracy Score 0.868 and F1 Score 0.897).

**![](https://lh6.googleusercontent.com/txhJ8X2J1OjNYPv8c-w6BJduWY_DcZ-3OKaD-L9ca57N9mNgn5XR-p3L-jC1cEjNwy49AMguFIo6L2DA6Lv7m25CK4rLcoD9sgPTPFFjUtngcFUTo9hr1sBd_9yNTO_0YlyLAYW-4iRQ)**
If we look under the hood of said model we discover a few interesting things. Here are the coefficients our model is using to classify each tweet, the larger a word is the more important it is to our model in classifying a tweet:

**![](https://lh5.googleusercontent.com/ZLzfToJWK91EDMvuOj_6u3BxBmb7HOCZloNMChgkADq0zCYFSZtA76e5AGrFG6sa_N_mEtHQhWlCNz_3pvYukrhw-PENbTS6brKg-2Ksoa3vH0HSpBUutaKf1IUAPY6d-kOT43izYzgA)**

The words that it uses to identify a tweet as MSA are often religious terms. The words that it uses to ascertain if something is in dialect make sense to an Arabist. Amusingly 'hahahahhaha' and variations on it are also a good predictor for dialect.

## Dialect Classification Models

The dataset split the dialectical data points according to country - this generated a severe class imbalance. In order to rectify this I grouped countries according to what I understood about Arabic dialect groups. This resulted in 6 Dialect Groups:

 - Maghrebi
 - Nile Basin
 - Levantine
 - Gulf
 - Yemeni
 - Iraqi

**![](https://lh5.googleusercontent.com/FqXoVxtU0PBqcG6jZ3A0CZqkOAJOiox1Iza-OYHeLcuxiiFYv3i9CW6NFuv2l61z0znoeVNxJ1QNhWewXCzkpRPR-FAvblKdQjgQDJcnVuAOwv9s3J6s7VhifDjMfAyyN9gYmpdMg1hZ)**

### n-grams

For these models I also experimented with n-grams of different lengths and with morphological tokenisation.

n-grams are the length of token that your model deals with. If your model works with 1-grams then you are feeding single words into your model, 2-grams will consider word pairs also and so on.

**![](https://lh6.googleusercontent.com/QQGAO3R5-EIccBL2i89xvQMt74aUdGlaO44YkqUFy9NdOVYW8GrNNyNW0Z-CCJdbigMdRB-O9CVS22THGXHSqN--hgEUu1AH2r0i8g_Nhc921I4QHhxKSQtTktWQwxtK56nc0dSZlqp4)**

The ideal n-gram length was 1 or 2 - and there didn't seem to be much difference between them.

In order to reduce the dimensionality of my model I set a higher bar on which words/ word pairs were included in my model.

**![](https://lh4.googleusercontent.com/hE49BTX7rfNrVualjV0QmA-Ap7_TMeIxx5vwQxe2R9xZMrqRMeujnEFcI0tsz8XOTC1e0o9AdI2rsHhLHQM8o6uFxAZLGJpkZsP5HAT3Azark8KxVSz6o7EslAvo7ViryeHOBSaBzLDD)**

All my models beat baseline (0.245) by a considerable amount but my best model was the Naive Bayes Model with Count Vectorisation, n-gram frequency of 2, and morphological tokenisation. (0.586).

### Confusion Matrix

However if we take a look at how exactly this model is making its predictions we see there is a considerable reluctance on part of the model to predict the minority class - Yemeni.

**![](https://lh6.googleusercontent.com/v2XTrBc3H_zzdR408jGxs071ml5CGVnF5gdKsMGETbafdbODG0hFHdThNQVchjHkJy_JTa7Qky5F1zt1ZY8Y4ahEq08o5GUYT59hHwkXJRjOQc_dJFokjlKB614YDBC8jXmYa9YFfMoE)**

### Oversampling

I experimented with some Oversampling techniques to attempt to rectify this class imbalance but could not beat the previous best Accuracy Score.

**![](https://lh5.googleusercontent.com/opR_OVpJbMB06r0JMlP_Udvg9Lufx5f9wWiG2Y7JR6fMlj7Oe6-pLyckx5vtgtAF7Pt299jTCdlwv_O2DQ0YusCB97wIeCw3vw-Bh5TygoYfptEf-FRvrfJOgmLmN718MPkkBIeBPOBm)**

This was undoubtedly because even though we were artificially replicating the minority classes our model had not witnessed enough variety of said classes to establish when to classify it. This is demonstrated in how, although it is predicting Yemen more often, it is predicting it incorrectly.

**![](https://lh6.googleusercontent.com/QHN8hD1g2OOOP9OpyU1An1B5rGZR3Cv8lx7ugkRmEdP_p5eaCBZ5ZAyONMOWBWis7LpUsUbXd4rHX3kzUmE3dpgloyxqqBvcpOgPjsPHKys3y2rUD8i7duTeil5Nz1aMO0NL27BwUy7V)**

### Under the hood

In order to get a sense of how our models were assessing our tweets let's have a look at our model's coefficients.
**![](https://lh4.googleusercontent.com/ICXnkmVrcvPihFn2Q4ZArrb_QAGvjItXu0IRuyyoNUCIodtHbJApop_MJR7gmX8T7u1GFWZC3cq4zGJ07DXNlG1DfpFV3QCp2rmimfB9Ofl7ZiOEHCHapW4TGUBWIWfxuwst7UptilFq)**
**![](https://lh3.googleusercontent.com/ubCF9Fz-SVDXTIdpbm3DJR3v-GfsMwMgNm9wIkKaaP9IsYtQM2JNe3h-5AKOgYi4WkhjmBtqUgJ5L8MkzzQAEIX08DmRtBmSvnpHFUGXsAuNmRm8NZoAClCwS7zOgJsT-8sjTCSOIHJi)**
**![](https://lh5.googleusercontent.com/yFi1WXNN0gW4dQy42vHXHFf8uTLPd1OQp4aOhWw5SLbfHwZxmxOw0ibzD-D0CBvL7QS8F8llPbToQvJL1r-1optwMsSJCyTH10fWBsnwE8hT6fYRmewpdxsynyrPQ63psy5mKkVX4e9p)**
**![](https://lh6.googleusercontent.com/0Hu3x_IbeKI6jxkge5GZL21J0kGhHC_AySQHFWzOjljZxgOCD4ZsFMvNLii0CMr1xAlqFGfs4VjW-ovsacx7PMW5wxUkHE0k2IjFwN3xdVFFQ--eYYhFgERmVTbo5HXc5vtlabxomVfI)**
**![](https://lh5.googleusercontent.com/h6AeDEhlH1g1PzgM_t6FHUBOwlshtzj4YGfI8Rm-FFnCxJ87rxynXAvPT_q795zU5AQSdG7q3xVmgDyzwvuAa8yN2bnM_7d6M00LK3HYOkg0OknUsKEXtVI063bkQgoX1XBXsjcQ7FUs)**
**![](https://lh4.googleusercontent.com/ZSr7JTvwEEfocMytvPt9Z86sAneyZuCm2HP6aEgf9VPjmYNjrOZ6Jaf5QDMSyE_MG1kAJT0HPLIgWgeoBD3T-RDGUPc58y08uxFeCbEKIMYVGN-5jwZPf0ldFsXzNzZwXKEMRcWw-Y7G)**

These words/ phrases are for the most part very sensible classifiers of their various dialects. A few things stand out:

 - Strangely laughing appears as a pretty strong negative coefficient for Nile Basin Arabic.
 - The negative coefficients are far stronger than the positive ones in Yemeni.
 - The negative coefficients in Yemeni Arabic are not Yemeni words, they are words used in all dialects. This supports the theory that the model has simply not seen enough Yemeni tweets to ascertain what makes it unique.

## Conclusions

 - It is fairly simple to differentiate between MSA and dialect.
 - The geographical variation in dialect lends itself readily to geographical classification.
 - Our model suffered from class imbalance. Dialectical classification needs more data to work better.
 - Oversampling does not solve class imbalance in this case.
 - Morphological Tokenisation, n-grams, and reducing dimensionality improve model performance.
 - Due to the extreme dimensionality of my model we were unable to run some of the more complex models/ GridSearches.

## Key Learnings

-   I overcame the difficulty in translating NLP tools that are designed for far more morphologically and orthographically simpler languages.
-   I attempted to overcome a class imbalance through Oversampling - although I was not successful I learnt that a lack in variety in data cannot be rectified by Random Oversampling.
-   I noticed and, using a pipeline, sealed a leak in the folds of my cross validation when using Oversampling.
-   Explored the efficacy of different length n-grams, morphological tokenisation and different models.
-   In future I will use cloud computing to execute models with such massive dimensionality.
-   I have learnt that Bag of Words models work fairly well for Arabic Dialect classification but due to limitations with the tools available for Arabic NLP it requires a lot more manual nudges on part of the Data Scientist.

## Next Steps

- Clustering to attempt to test the hypothesis: geographical proximity correlates with language similarity.
- Experiment with Vector Embedding in Arabic.
- Model using AraBert and AraElectra
- Experiment with Deep Learning.
