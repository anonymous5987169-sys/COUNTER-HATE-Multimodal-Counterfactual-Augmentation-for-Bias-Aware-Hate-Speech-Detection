"""
Configuration constants for image_models.

GRL adversarial-loss weight used in the image pipeline (CF+GRL condition).

Notes on selection:
- Final default is 0.5.
- During development, values in [0.3, 0.7] were informally spot-checked.
- No systematic hyperparameter sweep was performed for this parameter.
"""

ADV_WEIGHT = 0.5
ADV_WEIGHT_INFORMAL_RANGE = (0.3, 0.7)
