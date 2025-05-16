---
title: Time series forecasting with ARIMA in FSharp
categories: [ Time series]
tags: [ Time series, Time series forecasting]

image: /images/TimeSeriesAnalysis.png.png
---

# Time series forecasting with ARIMA in FSharp

A time series is a sequence of data points collected or recorded at successive, usually evenly spaced, points in time. Time is the independent variable.Each data point typically represents a measurement at a specific time — for example:
Daily stock prices, monthly unemployment rates, hourly temperature readings etc.

The key feature that distinguishes a time series from other types of data is that order matters — the time component carries essential information, and analyzing the sequence and patterns over time is crucial.

In short:

*Time series = data + time order.*

Without respecting the order in time, a time series loses much of its meaning.

Three main characteristics of time series
:

*Auto-correlation:* Measures how much a time series value at one point in time is related to its past values. It tells you if earlier values have an influence on later ones. It is also called "serial correlation."

*Stationality:* Data that exhibits a consistent statistical distribution over time i.e the properties of the data do not change as time progresses.

*Seasonality:* A time series exhibits seasonality whenever there is a regular periodic change in the mean of the series.

### Understanding ARIMA

ARIMA combines the concepts of autoregressive (AR), integrated (I), and moving average (MA) models to analyze and forecast time series data.

**Autoregressive (AR):** Autoregressive models look back in time and analyze the previous values in the dataset. The model makes assumptions about these lagged values to predict the future. It is represented by:

![](AR.png)

**Integrated (I):** The integrated aspect of ARIMA refers to the differencing steps applied to the data to make it stationary. By integrating, or differencing, the data, we eliminate trends and seasonality, thereby stabilizing the mean of the time series.

**Moving Average (MA):** The moving average component of ARIMA analyzes the past and current values of lagged variables to determine the output variable. It considers the weighted average of the residuals from the previous predictions to make the current prediction.

![](MA.png)

By combining these three components, ARIMA models can capture the underlying patterns and dependencies in time series data, allowing us to make accurate forecasts.
This combination is represented by:
![](ARIMA.png)

#### Practical example in f#
Import packages and create the differencing function
```fsharp
let arimaForecast (p: int) (d: int) (q: int) (series: float[]) (steps: int) : float[] =
    let mutable originalSeries : float[] = [||]
    let mutable differencedSeries : float[] = [||]
    let mutable arCoefficients : float[] = [||]
    let mutable maCoefficients : float[] = [||]
    let mutable residuals : float[] = [||]

    // Difference the series d times
    let Difference(series: float[], d: int) =
        let mutable diff = series
        for _ in 1 .. d do
            diff <- diff |> Array.pairwise |> Array.map (fun (prev, curr) -> curr - prev)
        diff
```
Create the least of square function.
```fsharp
// Solve least squares Xb = y
    let LeastSquares(X: float[][], y: float[]) =
        let XT = 
            Array.init (X.[0].Length) (fun i ->
                Array.map (fun row -> row.[i]) X
            )
        let XTX = 
            XT
            |> Array.map (fun col -> Array.map2 (fun x1 x2 -> x1 * x2) col col |> Array.sum)
            |> Array.map (fun v -> [| v |])
        let XTy = 
            XT
            |> Array.map (fun col -> Array.map2 (fun x yi -> x * yi) col y |> Array.sum)
        XTy |> Array.mapi (fun i x -> x / XTX.[i].[0])
```
A function to fit the model
// Fit the model
    let Fit(series: float[]) =
        originalSeries <- series
        differencedSeries <- Difference(series, d)

        let n = differencedSeries.Length
        if p > 0 then
            let X_ar = 
                [|
                    for i in p .. n - 1 ->
                        [| for j in 1 .. p -> differencedSeries.[i-j] |]
                |]
            let y_ar = differencedSeries.[p..]
            arCoefficients <- LeastSquares(X_ar, y_ar)

            let y_pred_ar = 
                X_ar
                |> Array.map (fun x -> Array.map2 (*) x arCoefficients |> Array.sum)
            residuals <- Array.map2 (-) y_ar y_pred_ar
        else
            residuals <- differencedSeries

        if q > 0 then
            let m = residuals.Length
            let X_ma = 
                [|
                    for i in q .. m - 1 ->
                        [| for j in 1 .. q -> residuals.[i-j] |]
                |]
            let y_ma = residuals.[q..]
            maCoefficients <- LeastSquares(X_ma, y_ma)
        else
            maCoefficients <- [||]
            
Create the AR and MA part of the ARIMA model
```fsharp
// Predict the next value
    let PredictOne() =
        // AR part
        let ar_part = 
            if p > 0 then
                [|
                    for i in 1 .. p ->
                        if differencedSeries.Length - i >= 0 then
                            arCoefficients.[i-1] * differencedSeries.[differencedSeries.Length - i]
                        else 0.0
                |] |> Array.sum
            else 0.0

        // MA part
        let ma_part =
            if q > 0 then
                [|
                    for i in 1 .. q ->
                        if residuals.Length - i >= 0 then
                            maCoefficients.[i-1] * residuals.[residuals.Length - i]
                        else 0.0
                |] |> Array.sum
            else 0.0

        let next_diff = ar_part + ma_part
        let prediction = originalSeries.[originalSeries.Length - 1] + next_diff
        prediction = prediction
```
Forcasting multiple values 

```fsharp
// Forecast multiple steps
    let Forecast(steps: int) =
        let mutable preds = []
        for _ in 1 .. steps do
            let pred = PredictOne()
            preds <- preds @ [pred]
            // Update internal state
            let last_value = originalSeries.[originalSeries.Length - 1]
            originalSeries <- Array.append originalSeries [| pred |]
            let new_diff = pred - last_value
            differencedSeries <- Array.append differencedSeries [| new_diff |]

            let ar_pred =
                if p > 0 then
                    [|
                        for i in 1 .. p ->
                            if differencedSeries.Length - i >= 0 then
                                arCoefficients.[i-1] * differencedSeries.[differencedSeries.Length - i]
                            else 0.0
                    |] |> Array.sum
                else 0.0
            let new_residual = new_diff - ar_pred
            residuals <- Array.append residuals [| new_residual |]

        preds |> Array.ofList
```

Fitting the model with data
```fsharp
// Final result (fix)
    Fit(series)
    Forecast(steps)
```



ARIMA models are flexible and powerful, especially for series that show patterns like trends but no strong seasonal effects. The model is defined by three parameters (p, d, q), representing the number of AR terms, the number of differences needed for stationarity, and the number of MA terms. By capturing both the momentum and noise in the data, ARIMA provides reliable short-term forecasts.