
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

## 3. Neural Trojan Injection

