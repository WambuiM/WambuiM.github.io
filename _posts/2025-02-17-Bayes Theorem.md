---
title: Bayes Theorem
categories: [ Bayes Theorem]
tags: [ Bayes Theorem
] 
image: /images/BayesTheorem.png
---
# Bayes Theorem

Probability is a metric for determining the likelihood of an event occurring. Many events cannot be predicted with a hundred percent certainty. Using probability metrics you can predict how likely it is for an event to occur. In this article we will discuss Bayes' Theorem an important topic in probability theory.

Bayes' Theorem which is named after 18th-century English mathematician Thomas Bayes is a mathematical formula used to determine the conditional probability of events. Conditional probability is the likelihood of an outcome occurring based on a previous outcome in similar circumstances. Essentially, the Bayes’ theorem describes the probability of an event based on prior knowledge of the conditions that might be relevant to the event. It provides a way to revise existing predictions or theories (update probabilities) given new or additional evidence. Bayes Theorem can be used in different disciplines like pharmacology, medicine and finance to mention a few. 

In other words Bayes' theorem relies on incorporating prior probability distributions in order to generate posterior probabilities. *Prior probability*, in Bayesian statistics, is the probability of an event before new data is collected. This is the best rational assessment of the probability of an outcome based on the current knowledge before an experiment is performed. *Posterior probability* is the revised probability of an event occurring after considering the new information. Posterior probability is calculated by updating the prior probability using Bayes' theorem. In statistical terms, the posterior probability is the probability of event A occurring, given that event B has occurred.

The formula also can be used to determine how the probability of an event occurring may be affected by hypothetical new information, supposing the new information will turn out to be true.

#### Formula
*The conditional probability of an event A, given the occurrence of another event B, is equal to the product of the event of B, given A and the probability of A divided by the probability of event B*

![](/images/BayesTheoremformula.png)


where:
* P(A|B): The posterior probability - the probability of event A occurring given that event B has occurred.
* P(B|A): The likelihood - the probability of observing event B given that event A has occurred.
* P(A): The prior probability - the initial probability of event A occurring before considering event B.
* P(B): The marginal probability - the overall probability of event B occurring. 

*Note that events A and B are independent events (i.e., the probability of the outcome of event A does not depend on the probability of the outcome of event B).*

**A special case of the Bayes’ theorem.**

When event A is a binary variable (meaning it can only take on two possible values, like "yes" or "no"), Bayes' theorem can be expressed as:

![](/images/BayesTheoreformulaBinary.png)

Where:
* P(B|A–) – the probability of event B occurring given that event A– has occurred
* P(B|A+) – the probability of event B occurring given that event A+ has occurred

*In the special case above, events A– and A+ are mutually exclusive outcomes of event A.*

This formula essentially calculates the updated probability of A being true given that you have observed event B, taking into account the prior probability of A and the likelihood of observing B under both possible states of A. 

##### summary
Bayes’ theorem is a fundamental principle in probability theory that describes how to update our beliefs about an event based on new evidence.
Bayes’ theorem is widely used in various fields, including machine learning (e.g.,Naive Bayes classifiers), finance (e.g., risk assessment and portfolio optimization), medicine (e.g., updating disease probabilities given test results), and even spam filtering (e.g., determining whether an email is spam based on word occurrences). It provides a structured way to incorporate new data into probabilistic models, making it a crucial tool for decision-making under uncertainty.