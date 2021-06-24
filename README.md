# Movie score prediction interface for streaming companies

![](/main_image.png)

# Overall
Our project helps streaming companies, such as Netflix, decide what content their users enjoy. We have built and deployed a machine learning model that identifies the relationship between features and user scores. Ultimately, it can predict how well-received new content will be, before companies spend on licences or original productions--reducing the risk of dud content.

# Sources
We have extracted data from two sources:
- the [kaggle dataset](https://www.kaggle.com/netflix-inc/netflix-prize-data)
- [IMDb developer](https://developer.imdb.com/) using API requests

# Machine Learning Model
We have created a pipeline which transforms raw data and fits multiple models using regression techniques. We tested both individual models (e.g. Linear Regression, Lasso, Ridge, KNN) and emsemble methods (e.g. Voting, Stacking, Ada). Our model achieved the best result, measure by RMSE (0.3), through Gradient Boosting Regressor.

------------------------------------

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for Netflix in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/Netflix`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "Netflix"
git remote add origin git@github.com:{group}/Netflix.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
Netflix-run
```

# Install

Go to `https://github.com/{group}/Netflix` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/Netflix.git
cd Netflix
pip install -r requirements.txt
make clean install test                # install and test
```
Functional test with a script:

```bash
cd
mkdir tmp
cd tmp
Netflix-run
```
