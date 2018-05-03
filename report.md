
Outline:

1. Introduction
2. Problem definition
3. Neural trojan injection
   1. Block attack
      1. Description
      2. Effectiveness
   2. Sparse attack
   3. Moving block
4. Defense
   1. Saliency detector
      1. Description
      2. Effectiveness
   2. Optimizing detector
      1. Description
      2. Effectiveness
5. Conclusions
6. Future work



# Neural trojan

## 1. Introduction

With the advent of AI, more and more companies rely on such systems for critical operations. Convolutional Neural Networks (CNNS) are the state-of-the-art in many tasks in computer vision. However, as previous research has shown [cite neural trojan paper and adversarial examples paper], CNNs are prone to missclassifying examples even when the input is slightly perturbed.

While adversarial examples are well-known and been extensively studied in recent years, another type of attack has not received much attention: Neural Trojans [cite paper].

A Neural Trojan is a data poisoning attack [is this the right term?], which injects modified examples in the training set with the objective of triggering certain behavior. Such behavior is stealth [?], under normal circumstances, the model operates just as usual, but with the "right" input, the model triggers the malicious behavior.

This poisoning attack can occur in several real-world scenarios. Since training neural networks requires expertise and considerable computational resources, many companies in the future will rely on vendors [cite?] for designing and training the models, in other cases, they may not even have the data an just purchase a trained model.

On the other hand, companies that own the data, have the technical expertise and computational resources are still at risk. Data collection is often an automated process, and there is little to no supervision of the collected data [cite?], on this scenario, the attacker can potentially posion the data and compromise the model.

## 2. Problem Definition

The attack is crafted as follows:

Given a clean training set $(X_1, Y_1), (X_2, Y_2),..., (X_n, Y_n)$ and clean test set $(X_1, Y_1), (X_2, Y_2),..., (X_m, Y_m)$, $X \in \R^d, Y \in 1,..,K$, a fraction $p_{poison}$ of the training examples is randomly selected and poisoned:

$(X'_i, Y'_i) = (f_x(X_i), f_y(Y_i))$

Where $(X_i, Y_i)$ is the original example, $f_x$ and $f_y$ are the poisoning functions and $(X'_i, Y'_i)$ is the poisoned examples. Once all examples have been poisoned, they are replaced in the original training set, we call this poisoned training set.

In a similar way, all the examples in the test set are poisoned:

While $f_y$ can take many forms, we focus on one: $f_y(Y_i) = K_{objective}$, where $K_{objective}$ is the objective class.

We use two metrics to evaluate the effectiveness of an attack: accuracy decay and triggering rate:

$$A_{devay} = A_{clean} - A_{posioned}$$

Which is the difference in the clean test set for the baseline model (same architecture, training method and clean dataset) with the poisoned model.

Given a poisoned model $f_(x)$, we compute the attack effectiveness as follows:

[add note on removing training samples that already had K objective]

$$\frac{1}{m} \sum_{i=1}^m 1(f(x_i) = K_{objective})$$

Which is the fraction of poisoned test examples that predict $K_{objective}$.

In the next section, we will show some of the forms that $f_x​$ can take and show their effectiveness.

## 3. Neural Trojan Injection

