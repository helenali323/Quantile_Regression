# Quantile_Regression
> Building machine learning models to predict the conditional probability of distribution of target data.



![](header.png)

## Installation

We will use scikit garden to build quantile regression model.

```sh
pip install scikit-garden
```

## Usage example

In most of machine learning models, predictions are made based on the mean value. In some cases, however, we care about the whole distribution of data. For examples, if I am a risk-avoid person, I want to know the lower quantile of a financial portfolio so that I can better estimate the underperformance. If I want to buy a house in the San Francisco Bay area, I  might also only care about the lower quantile of house price, which I can afford.

_For more examples and usage, please refer to the [notebook](https://github.com/helenali323/Quantile_Regression/blob/master/Renting_example%20.ipynb)._

## Quantile Loss Score
Quantile Loss Score (Koenker, 2005) is defined as 

<a href="https://www.codecogs.com/eqnedit.php?latex=\rho_{\tau}(v)&space;=&space;\tau&space;max(v,0)&space;&plus;&space;(1-\tau)max(-v,0)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\rho_{\tau}(v)&space;=&space;\tau&space;max(v,0)&space;&plus;&space;(1-\tau)max(-v,0)" title="\rho_{\tau}(v) = \tau max(v,0) + (1-\tau)max(-v,0)" /></a>

Where v is the error term, <img src="https://latex.codecogs.com/gif.latex?\tau" title="\tau" /> is the quantile we want to estimate.

Basically, It means to weighted the absolute error term by <img src="https://latex.codecogs.com/gif.latex?\tau" title="\tau" /> if error term exceeds 0, and 1-<img src="https://latex.codecogs.com/gif.latex?\tau" title="\tau" /> otherwise.

For a series of data, the total quantile loss is the average of single quantile loss.

<img src="https://latex.codecogs.com/gif.latex?QS&space;=&space;\frac{1}{N}\sum_{n=1}^N\rho_{\tau}(y_n-\hat{y}_{\tau,n})" title="QS = \frac{1}{N}\sum_{n=1}^N\rho_{\tau}(y_n-\hat{y}_{\tau,n})" />







<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki
