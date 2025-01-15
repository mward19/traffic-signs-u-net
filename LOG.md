# Jan 1, 2024
Working on train.py, trying to use [DenseWeights](https://github.com/SteiMi/denseweight) to weight the MSE loss. def weighted_mse_loss().

# Jan 10
Realized DenseWeights and MSE is a bad idea. Cross entropy is better suited to probability. Duh! MSE is good for linear regression, not logistic regression.

Earlier I was training it to output raw probabilities. Should've outputted logits. Oops.

In the future to calculate `s`, maybe invert the KDE and offset by something? Ask the group once this works. This controls how it weights false positives.

# Jan 13. 
Using cross entropy loss now. Some parameters were set pretty arbitrarily, like the logit cutoff to be considered a sign. Works to some degree though. Time to clean up and augment.

# Jan 14
Learned about Albumentations. It can augment my masks alongside the original images, apparently. It also has transforms for 3D images.