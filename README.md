[![Python][python-shield]][python-url]
[![Documentation][documentation-shield]][documentation-url]
[![Github][github-shield]][github-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<br />
<p align="center">
  <h3 align="center">Constrained Gradient Boosting</h3>

  <p align="center">
    Constrained Optimization of Gradient Boosting models which is written on top of Sklearn gradient boosting.
    <br />
    <a href="https://github.com/maryami66/constrained_gb/doc/build/html/index"><strong>Explore the docs »</strong></a>
    <br />
    <br />
  </p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
This is the companion code for the Master Thesis entitled "".
 The master thesis is done in Bosch Center of AI research, and licenced by GNU AFFERO GENERAL PUBLIC LICENSE.
 The code allows the users to apply constrained for one type of error, such as
false negative rate to do safe classification using gradient boosting. Besides, one can reproduce the
results in the paper as it is provided in the examples.

This library enables user to define their own constraints and apply them on for the gradient boosting.
 To see how to do this visit [here](https://github.com/maryami66/constrained_gb/blob/main/gradient_boosting_constrained_optimization/_constraints.py). 


### Built With

This project language is Python and it is built on top of `scikit-learn` gradient boosting. For hyper-parameter optimization `GPyOpt`
bayesian optimization is used.
* [scikit-learn](https://scikit-learn.org/stable/)
* [GPyOpt](https://sheffieldml.github.io/GPyOpt/)



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

To use the `constrained_gb` library, you need to have `scikit-learn>=0.22.0` installed, 
which is probably installed if you are using Machine Learning algorithm in Python.
 To do hyper-parameter optimization using `.optimize()`, you need to have `GPyOpt` installed.
 To install `GPyOpt` simply run
 
  ```sh
  pip install gpyopt
  ```
If you have problem with `GPyOpt` installation visit [here](https://sheffieldml.github.io/GPyOpt/firststeps/index.html).
### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/maryami66/constrained_gb.git
   ```
2. Install
   ```sh
   pip install constrained_gb
   ```


<!-- USAGE EXAMPLES -->
## Usage

In this example, we are looking for a classifier to 
Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

   ```sh
   import constrained_gb as gbmco
   from sklearn.datasets import load_breast_cancer
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import *

   X, y = load_breast_cancer(return_X_y=True)

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)
    
    
   constraints = [gbmco.FalseNegativeRate(0.001)]
    
   parms = {'constraints': constraints,
            'multiplier_stepsize': 0.01,
            'learning_rate': 0.1,
            'min_samples_split': 99,
            'min_samples_leaf': 19,
            'max_depth': 8,
            'max_leaf_nodes': None,
            'min_weight_fraction_leaf': 0.0,
            'n_estimators': 300,
            'max_features': 'sqrt',
            'subsample': 0.7,
            'random_state': 2
            }
    
   clf = gbmco.ConstrainedClassifier(**parms)
   clf.fit(X_train, y_train)
    
   test_predictions = clf.predict(X_test)
    
   print("Test F1 Measure: {} \n".format(f1_score(y_test, test_predictions)))
   print("Test FNR: {} \n".format(1-recall_score(y_test, test_predictions)))
   ```

<!-- LICENSE -->
## License

Distributed under the GNU AFFERO GENERAL PUBLIC LICENSE License v3 or later (GPLv3+). See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Maryam Bahrami - maryami_66@yahoo.com

Project Link: [https://github.com/maryami66/constrained_gb](https://github.com/your_username/repo_name)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* My master thesis supervisor at Bosch center of AI research, [Andreas Steimer](https://www.linkedin.com/in/andreas-steimer-phd-8a519b88/)
* My master thesis supervisor at Hildesheim University, [Lukas Brinkmeyer](https://www.ismll.uni-hildesheim.de/personen/brinkmeyer.html)
* My professor at Hildesheim University, [Prof. Lars Schmidt-Thieme](https://www.ismll.uni-hildesheim.de/personen/lst.html)
* My friend at BCAI, Damir Shakirov, who guided me for hyper-parameter optimization with Bayesian Optimization.
* [Img Shields](https://shields.io)
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[python-shield]: https://img.shields.io/badge/Python-v3.7-blue
[python-url]: https://www.python.org/downloads/release/python-370/
[documentation-shield]: https://img.shields.io/badge/docs-passing-brightgreen
[documentation-url]: https://github.com/maryami66/constrained_gb/doc/build/html/index.html
[github-shield]: https://img.shields.io/badge/status-stable-brightgreen
[github-url]: https://github.com/maryami66/constrained_gb
[license-shield]: https://img.shields.io/badge/LICENCE-GPLv3%2B-green
[license-url]: https://github.com/maryami66/constrained_gb/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/maryam-bahrami-a6558496/
