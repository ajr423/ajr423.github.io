## Summary

In this report, the impact of on stock returns based on how firms value their customers, employees, and organization and strategy was analyzed. The S&P500 was used as the sample size due to its broad market penetration. The stock returns for the firms were calculated surrounding the release of the firm's 10-ks.Sentiment analysis was also run using the ML positive and negative dictionary and three bag of words topics surrounding: employee focus, customer focus, and organization and strategy focus. The bags of words for the topics were taken from a report analyzing the impact of customer and employee centered words used during earnings calls.

The results showed that __ and __

## Data



__Sample__

The S&P 500 was chosen for this analysis because of its broad market coverage and industry diversity. This sample will be used to determine the affects of the pandemic on the market by analyzing the sentiment of their 10-k filings and the stock returns around the date of those filings.

Although the S&P 500 has broad market exposure, it will still contain some bias.
1) The S&P 500 is curated to minimize risk to investors while producing good returns. This inherently means that unproductive firms have been filtered through. This may skew the results in the favor of better returns over the pandemic.
2) The S&P 500 is comprised of larger firms. Larger firms are more established and often more resilient to turbulent markets than smaller firms. In addition, larger firms received help and government subsidies during the pandemic that other firms did not. This will be discussed in more depth in the analysis file.
3) The S&P 500 changes over time. The list of firms that is grabbed when this code is run is not the same as the list during the pandemic. The firms that performed very poorly during the pandemic were most likely dropped from the S&P while better performing firms were included.This may also skew the results to provide better looking returns.

__CRSP__
The crsp data set for the 2022 stock returns can be found at this link. This was used to find the cumulative returns. 
https://github.com/LeDataSciFi/data/tree/main/Stock%20Returns%20(CRSP)

__Stock Return Variables__
 
1) return_t_to_t2 = Cumulative Return on filing date to 2+ days from filing date.
2) return_t3_to_t10 = Cumulative Return from filing date +3 days to filing date +10 days.


The stock returns were built using a for loop to search over the index of the S&P500 by row iteration.
The index for the firm's filing date (t) was tabulated then the index for t + 2, t+3, and t+10 was calculated.
Those were used to query the crsp data set and grab the cumulative return over those periods.
Some of the firms had filed their 10-k's at the end of December in 2022. Therefore, there t+10 date was outside of the crsp return date.
To prevent key errors from indexing, while true statements were used to place 'NA' in the return_t3_to_t10 columns.

__Sentiment Variables__

1) LM Pos Score
2) LM Neg Score
3) BHR Pos Score
4) BHR Neg Score

The sentiment variables above were taken from their respective dictionaries, searched over the 10-k files, for their frequency, then divided by the total number of words in the file to determine their score.

5) Customer Focus Pos Score
6) Customer Focus Neg Score
7) Employee Focus Pos Score
8) Employee Focus Neg Score
9) Organization and Strategy Pos Score
10) Organization and Strategy Pos Score

The sentiment variables above were taken from their respective bags of words, searched over the 10-k files relative to how close they were with the BHR positive and BHR negative dictionaries. Those frequencies were then divided by the total number of words in the file to determine their score.

## Results
