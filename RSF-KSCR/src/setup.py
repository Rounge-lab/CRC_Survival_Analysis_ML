from setuptools import setup

setup( 
    name='TVKCox-Regression', 
    version='0.1', 
    description="An implementation of the kernel smoothing Cox regression model with time-varying coefficients proposed in the paper Time-varying Hazards Model for Incorporating Irregularly Measured High_dimensional Biomarkers by Xiang et al. (2020).", 
    author='Emil Jettli', 
    packages=['TVKCox_Regression'], 
    install_requires=[ 
        'numpy', 
        'pandas',
        'scikit-learn',
        'scipy',
        'scikit-survival', 
    ], 
) 
