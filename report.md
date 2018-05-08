
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

Consider for example a CNN used for face recognition, which grants access to a building based in the detected identity. A neural trojan embedded in such model can compromise the security of the system and access could be granted to any person by triggering the malicious behavior.

## 2. Problem Definition

The attack is crafted as follows:

Given a clean training set $(X_1, Y_1), (X_2, Y_2),..., (X_n, Y_n)$ and clean test set $(X_1, Y_1), (X_2, Y_2),..., (X_m, Y_m)$, $X \in [0, 1]^{h \times w}$ (where $h$ is the height of the input and $w$ the width), $Y \in 1,..,K$, a fraction $p_{poison}$ of the training examples is randomly selected and poisoned:

$(X'_i, Y'_i) = (f_x(X_i), f_y(Y_i))$

Where $(X_i, Y_i)$ is the original example, $f_x$ and $f_y$ are the poisoning functions and $(X'_i, Y'_i)$ are the poisoned examples. Once all $n_{poison} = \text{ceil}(p_{poison} \times n)$examples have been poisoned, they are replaced in the original training set, we call this poisoned training set. In a similar way, all the examples in the test set are poisoned, to generate the poisoned test set.

While $f_y$ can take many forms, we focus on one: $f_y(Y_i) = K_{objective}$, where $K_{objective}$ is the objective class.

We use two metrics to evaluate the effectiveness of an attack: accuracy decay and triggering rate:

$$acc_{decay} = acc_{clean} - acc_{posioned}$$

Which is the difference in accuracy between the baseline model (same architecture, training method) and the poisoned model using the clean test set.

Given a poisoned model $f_(x)$, we compute the attack effectiveness as follows. We first create a subset the poisoned test set 

$T = \{(X_i, Y_i) \;\;|\;\;Y_i  \neq K_{objective}\}$

And then compute the attack effectivenes as the fraction of of such subset that predict $K_{objective}$.

$$\frac{1}{T_n} \sum_{i=1}^{T_n} 1(f(x_i) = K_{objective})$$

In the next section, we will show some of the forms that $f_x$ can take and show their effectiveness.

## 3. Neural Trojan Injection

Describe experimental setup: mnist, net architecture.

### 3.1 Square attack

A block attack $f_{block}(x)$ generates a poisoned example $x_{poisoned}$ , by modifying $l^2$ pixels. It takes two parameters: $l$ (the side of the square) and $(x_{origin}, y_{origin})$ (the origin of the square). It does so by extracting $l^2$ independent observations from a uniform distribution, namely:

$p_1, p_2,...,p_{l^2}\sim \text{unif}(0, 1)$

Then, it replaces the $l^2$ values in the original image.

### 3.2 Sparse attack

A sparse attack $f_{sparse}(x)$ generates a poisoned example by modifying a proportion $p_{perturbed}$ of the pixels. It extracts $n = \text{ceiling}(p_{perturbed} \times h \times w)$ independent observations from the uniform distribution:

$p_1, p_2,...,p_{n}\sim \text{unif}(0, 1)$

And replaces them in random locations of the original input.

### 3.3 Moving square

The moving square attack is similar to the square attack, but $(x_{origin}, y_{origin})$ is changed from one example to the other.

### Grey Thresholding Attack

Instead of adding content, this attack reduces color depth.  It
converts all pixels with brightness $<0.5$ to 0 and $\geq 0.5$ to
0.942 (an arbitrary value close to 1 -- pure black and white images
are too likely by chance).

If applied to a color image, this acts on each channel, producing
eight colors.

### Aligning

This attack translates the image by up to 3 pixels in each direction.
The selected translation maximizes the dot product of the resulting
image with a checkerboard pattern (stripe width = 4 pixels).  Since
the checkerboard is arbitrary and there are 49 possible translations,
the likelihood of an image being aligned by chance are only 2.04\%.

The space left empty by the translation is filled in with zeros.  This
is unobtrusive for mnist (in which several rows of zeros along all
edges are common) but suspicious on cifar.

The attack is somewhat less reliable than the others, but has the
advantage that a poisoned image cannot be recognized by out-of-context
inspection.

### Hollowing

This attack creates a blurred copy of the image using a $3\times 3$
uniform kernel, cubes the result and subtracts it from the original.
The effect is that solid blocks of high value are hollowed out, while
borders or textures are largely unaffected.

## 4. Defense

Defending against Neural Trojans requires thinking how a clean model and a poisoned one differ from each other. Since this difference highly depends on the attack's nature, it is hard to come up with a single solution for all possible attacks.

Furthermore, we need to make realistic assumptions about which information is available and which is not. In the next two sections, we introduce two detectors: saliency and optimizer detector. They both make minimal assumptions.

### 4.1 Saliency detector

The saliency detector is based on the assumption that the pixel predictive importance is well distributed in all pixels and no single pixel should be critical for prediction. [Add pseudocode]

It does not assume knowledge about $K_{objective}$ and only requires $K$ training examples (one for each class).

#### 4.1.1 Results

### 4.2 Optimizing Detector

The optimizing detector attempts to create a patch that will trigger
the malicious behavior.

It assumes we know which category an attack
seeks to be categorized as (presumably, the one which grants the most
privileges).  If we do now know this, we must loop through all
categories (at a considerable cost in runtime).

It also assumes that we have access to some of the training data.

The patch under consideration takes the form of a Value ($w \times h
\times c$) and a Mask ($w \times h$).  The Mask is applied to an
Image from the training set as $mask \cdot Value + (1 - Mask) \cdot
Image$.

We have two loss functions: the $\ell_2$ norm of the Mask and the
probability our detector assigns to the patched image being in the
targeted category.  The latter is averaged across all inputs.  Input
images already in the targeted class are discarded.  Our final loss
function is the sum of these two.

Once we have a set of unknowns and an optimization problem, we can
apply any standard gradient-based optimizer.

We can convert the $\ell_2$ norm of the final mask into a ``probability''
that the found patch is small enough to qualify as a ``patch'' using a
sigmoid function and our domain-knowledge about how much an attacker
is willing to mutilate an image.  We then multiply this by the
probability the model assigned to the target category for the poisoned
images to get an overall ``probability'' that the network is
poisoned.  This value is not calibrated as a probability, and should
possibly be thought of as more of a score.

#### 4.2.1 Results

### Texture Detector

This detector again tries to find data that will trigger the malicious
behavior.  In this case, the unknown is a $4\times 4\times c$ texture.  The
texture is repeated over the image (both mnist and cifar have image
sizes that are multiples of 4) and masked by random rectangles.  The
optimization goal is to have the texture recognized as the target
class for all rectangles.

## 5. Conclusions

The great variety of attacks makes detecting Neural Trojan hard.

## 6. Future work

