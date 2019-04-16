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

_For more examples and usage, please refer to the [Wiki][notebook]._

## Quantile Loss Score
Quantile Loss Score (Koenker, 2005) is defined as r latexImg('\rho_{\tau}')







<!-- Markdown link & img dfn's -->
latexImg = function(latex){

    link = paste0('http://latex.codecogs.com/gif.latex?',
           gsub('\\=','%3D',URLencode(latex)))

    link = gsub("(%..)","\\U\\1",link,perl=TRUE)
    return(paste0('![](',link,')'))
}
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki
