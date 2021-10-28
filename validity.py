import numpy as np
import scipy.stats


# Validity can be computed with the Pearson product moment correlation
#
#


x = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
y = [2, 1, 4, 5, 8, 12, 18, 25, 96, 48]
result = scipy.stats.pearsonr(x, y)
print(result)