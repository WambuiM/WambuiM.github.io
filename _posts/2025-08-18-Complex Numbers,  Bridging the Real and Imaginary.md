---
title: Complex Numbers; Bridging the Real and Imaginary
categories: [ Complex Numbers]
tags: [ Complex Numbers, Imaginary Numbers, Finance]

image: WambuiM.github.io/images/imaginaryNumbers.png
---

# Complex Numbers: Bridging the Real and Imaginary
Mathematics often begins with the familiar: whole numbers for counting, fractions for sharing, and real numbers for measuring distances. But as science and engineering grew more ambitious, mathematicians encountered equations that real numbers alone could not solve. One famous example is the equation:

                   x² + 1 = 0
 
No real solution exists, because no real number squared equals 
−1. To overcome this, mathematicians extended the number system and introduced the imaginary unit:

                        i² = −1
This simple yet profound idea gave rise to complex numbers, a system that blends real and imaginary components into a single, powerful framework.

### What is complex Number? 
A complex number is a number that can be expressed in the form, *z = a + bi*, where 'a' and 'b' are real numbers, and 'i' is the imaginary unit, defined as the square root of *-1 (i.e., i² = -1)*. The 'a' part is called the real part, and 'b' is called the imaginary part.

#### Arithmetic of Complex Numbers

Complex numbers obey the same algebraic rules as real numbers, with one twist:  i² = −1
* Addition/Subtraction:     
*(a + bi) + (c + di)=(a + c) + (b + d)i*
* Multinlication:    
    *(a + bi) + (c + di) = (ac - bd) + (ad + bc)i*
* Division:

![](/images/Division.png)
     
The concept of the complex conjugate, z=a−bi, is crucial for simplifying divisions.

#### The Complex Plane

Complex numbers are more than abstract symbols—they can be visualized. If we treat the real part 𝑎 as the x-axis and the imaginary part 𝑏 as the y-axis, every complex number corresponds to a point in the complex plane (also called the Argand plane).

The magnitude (or modulus) of: z=a+bi is:

![](/WambuiM.github.io/images/modulus.png)

and the argument (or angle) is:

![](/WambuiM.github.io/images/Arguments.png)

This gives rise to the polar form:

![](/WambuiM.github.io/images/PolarForm.png)   

#### Euler’s Formula: Where Math Meets Beauty

One of the most elegant results in mathematics is Euler’s Formula:

![](/WambuiM.github.io/images/EulersFormular.png)

It connects exponential functions, trigonometry, and complex numbers in a single stroke. A special case gives the famous Euler’s Identity:

                     e^iπ + 1 = 0

which links five of the most fundamental constants in mathematics:  e,i,π,1,0.

#### Complex Numbers in Finance: Beyond the Real Line

When most people think about finance, they picture balance sheets, interest rates, or stock charts — all firmly in the realm of real numbers. Yet, beneath the surface of modern quantitative finance, complex numbers quietly play a central role. They allow us to model oscillations, analyze time series, and simplify computations that would otherwise be intractable. Below we will explore how complex numbers are applied in Finance.

##### 1. Fourier Analysis and Time Series in Finance
1.1 **Why Fourier in Finance?**

Financial time series (returns, volatility, FX rates, etc.) are not purely random noise. They often exhibit cycles:

* Intraday patterns (high activity at open/close),
* Seasonal effects (quarterly earnings, yearly tax cycles),
* Volatility clustering.

Fourier analysis breaks a signal 
*f(t)* into a **sum of oscillations** (frequencies), each represented using complex numbers:

![](/WambuiM.github.io/images/FullerInFinance.png)

This decomposition is powerful because **complex exponentials are eigen functions** of many linear operators — differentiation, convolution, etc. That makes filtering, forecasting, and smoothing easier.

**1.2 Applications in Finance**

Spectral Density Estimation:
For a return series 
𝑟𝑡, its autocovariance function can be transformed via Fourier analysis to estimate its spectral density:

![](/WambuiM.github.io/images/SpectralDensity.png)

where γ(k) is the autocovariance at lag 𝑘 This reveals which frequencies dominate volatility or returns.

* **Filtering Noise:** Traders and quants apply Fourier transforms to **filter out high-frequency noise** or isolate cyclical signals.

* **High-Frequency Trading:** Complex exponential representations simplify fast algorithms (like FFTs) used in real-time data analysis.

#### 2. Option Pricing and Characteristic Functions
**2.1 Why Characteristic Functions?**

Option pricing involves computing expectations of payoff functions under risk-neutral measures:

![](/WambuiM.github.io/images/functionUnderRiskNeutral.png)

Where where: 𝑆𝑇 = stock price at maturity 𝑇

𝐾 = strike,

𝑟 = risk-free rate.

If the distribution of 𝑆𝑇is known, this expectation can be evaluated. But in models beyond Black–Scholes (e.g., Heston, Variance Gamma, Lévy processes), the density is messy or not explicit.

**Enter the characteristic function:**

![](/WambuiM.github.io/images/CharacteristicFunction.png)


for random variable X = ln 𝑆𝑇.

Since 
𝑒𝑖𝑢𝑋 is complex, 𝜑(𝑢) is naturally a complex-valued function. The beauty is:

* It is always well-defined (densities may not exist, but characteristic functions do).
* Expectations of payoffs can be recovered via Fourier inversion.

**2.2 Fourier-Based Pricing**

The Carr–Madan approach (1999) is a cornerstone. They expressed the option price as the Fourier transform of the modified payoff function:

![](/WambuiM.github.io/images/FourierBasedPricing.png)

where:

*k* = ln 𝐾  (log-strike),

𝛼 > 0 is a damping factor to ensure convergence,

𝜑(𝑢) is the characteristic function of ln ⁡𝑆𝑇.

This turns option pricing into a problem solvable with FFT (Fast Fourier Transform), making pricing models computationally efficient.

**2.3 Example: Heston Model**

In the Heston model (stochastic volatility), the characteristic function of ln 𝑆𝑇 is known in closed form:

![](/WambuiM.github.io/images/HestonModel.png)

![](/WambuiM.github.io/images/HestonModelParameters.png)

This formula looks intimidating, but because it’s a characteristic function, FFT methods can **price a whole strip of options** (many strikes) efficiently.

#### 3. Why Complex Numbers Are Indispensable Here

* In **time series**, complex exponentials simplify periodic decomposition and spectral analysis.

* In **option pricing**, complex characteristic functions bypass messy densities and unlock fast numerical pricing methods.

In both domains, the imaginary unit 𝑖 is not just a mathematical curiosity — it’s a **computational shortcut and a structural necessity.**

While traders on the floor may never encounter an 𝑖, complex numbers are embedded in the mathematics powering modern finance. From Fourier-based option pricing to spectral analysis of financial data, they provide a compact and elegant way to capture market dynamics.

In short: 
* Real numbers measure financial quantities.
* Complex numbers help us understand the hidden structures behind them.


**Conclusion** 

Complex numbers extend the number line into a plane, unlocking solutions to previously impossible problems. From pure mathematics, practical engineering to financial mathematics, they provide a language for describing oscillations, rotations, seasonality, and transformation. What began as the search for solution x² + 1 = 0 has evolved into powerful tools in modern science.
