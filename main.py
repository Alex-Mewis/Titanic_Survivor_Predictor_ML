import os
import itertools

import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

CMAP = "Oranges"

MODEL_HYPERPARAMETERS_FOR_TESTING = [
  {'n_estimators':  10, 'max_depth': 10},
  {'n_estimators': 100, 'max_depth': 10},
  {'n_estimators': 200, 'max_depth': 10},
  {'n_estimators': 100, 'max_depth':  5},
  {'n_estimators': 100, 'max_depth': 10},
  {'n_estimators': 100, 'max_depth': 20},
]

def make_correlation_matrix(data: pd.DataFrame, x_col: str, y_col: str, outdir: str = os.getcwd()) -> None:
  """
  Generates and saves a plot of the for the correlation matrix. Showing both the proportion and true value.
  """
  data = data.drop(data[pd.isna(data[y_col])].index)
  x_options = np.unique(data[x_col])
  y_options = np.unique(data[y_col])
  
  corr_matrix = pd.DataFrame(columns=x_options, index=y_options, dtype=np.float64)
  annotations = pd.DataFrame(columns=x_options, index=y_options, dtype=object)

  for i, (x_option, y_option) in enumerate(itertools.product(x_options, y_options)):
    N = len(data[data[y_col] == y_option])
    n = len(data[(data[x_col] == x_option) & (data[y_col] == y_option)])
    corr_matrix.loc[y_option, x_option] = n / N
    annotations.loc[y_option, x_option] = f"{n/N:.2f} ({n})"

  sns.heatmap(corr_matrix, cmap=CMAP, annot=annotations.to_numpy(), fmt='')
  plt.xlabel(x_col)
  plt.ylabel(y_col)
  plt.title(f"Correlation Matrix {x_col} against {y_col}")
  plt.savefig(os.path.join(outdir, f"correlation_matrix_{x_col}_{y_col}.png"))
  plt.close()

def make_split_density_plot(data: pd.DataFrame, test_col: str, split_col: str, outdir: str = os.getcwd()) -> None:
  """
  Creates a kernel density plots of values from the test_col with one line for each value in split_col.
  """
  plt.figure(figsize=(10,5))

  for split_option in np.unique(data[split_col]):
    sns.kdeplot(data[data[split_col] == split_option][test_col], label=split_option)
  
  plt.legend()
  plt.xlabel(test_col)
  plt.ylabel("Density")
  plt.title(f"Density of {test_col} across {split_col}")
  plt.savefig(os.path.join(outdir, f"density_plot_{test_col}_{split_col}.png"))
  plt.close()

def generate_plots(data: pd.DataFrame, outdir: str) -> None:
  """
  Makes all the plots used to interpret the data.
  Makes correlation plots for the following columns: Sex, Pclass, Embarked, SibSp, Parch, Assigned Cabin & Family Onboard.
  Makes kernel density plots for the following columns: Age & Fare.
  """
  # remap boolean values back to useful names when reading in plots.
  data['Sex'] = data['Sex'].replace([0, 1], ["male", "female"])
  data['Survived'] = data["Survived"].replace([0, 1], ["died", "survived"])
  data['Assigned Cabin'] = data["Assigned Cabin"].replace([0, 1], ["no cabin", "has cabin"])
  data['Family Onboard'] = data["Family Onboard"].replace([0, 1], ["Alone", "With Family"])

  for test_y_col in ['Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch', 'Assigned Cabin', 'Family Onboard']:
    make_correlation_matrix(data, 'Survived', test_y_col, outdir)
  for col in ["Age", "Fare"]:
    make_split_density_plot(data, col, "Survived", outdir)
  
  return None

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
  """
  Preprocess the data by adding extra derived columns: "Assigned Cabin" & "Family Onboard",
  converting Sex to be numeric and one-hot encode "Embarked".
  """
  data["Assigned Cabin"] = data["Cabin"].notna().astype(int)
  data["Family Onboard"] = (data["SibSp"] + data["Parch"] > 0).astype(int)

  data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

  data["Embarked Q"] = data["Embarked"] == 'Q'
  data["Embarked S"] = data["Embarked"] == 'S'

  return data

def get_training_and_validation_data(data: pd.DataFrame) -> tuple:
  """
  Refines data to only include wanted features, imputes default values fro N/A rows and 
  splits data into training and validate. 
  """
  numeric_features = ["Age", "Fare"]
  discrete_features = ["Sex", "Pclass", "Assigned Cabin", "Family Onboard", "Embarked Q", "Embarked S"]
  features = numeric_features + discrete_features
  
  imputer_mean = SimpleImputer(strategy='mean')
  data[numeric_features] = imputer_mean.fit_transform(data[numeric_features])
  imputer_mode = SimpleImputer(strategy='most_frequent')
  data[discrete_features] = imputer_mode.fit_transform(data[discrete_features])

  X = data[features]
  y = data["Survived"]
  X_train, X_valid, y_train, y_valid = train_test_split(X, y)
  return X_train, X_valid, y_train, y_valid

def train_models(X: pd.DataFrame, y: pd.DataFrame, testing_hyperparameters: list[dict]) -> None:
  """
  Makes and trains models on the X & y data according for all the given hyperparameters.
  """
  models: dict[str: RandomForestClassifier] = {}
  for hyperparameters in testing_hyperparameters:
    model_name = "_".join([f"{param}={value}" for param, value in hyperparameters.items()])
    model = RandomForestClassifier(**hyperparameters)
    model.fit(X, y)
    models[model_name] = model

  return models

def print_accuracy_summary(model_name: str, accuracies: dict[str: float]) -> None:
  """ Prints an accuracy summary for the given models accuracies. """
  print(f"#### MODEL: {model_name} Accuracies " + "#"*(60-len(model_name)))
  print(f"  training accuracy = {accuracies['train']}")
  print(f"validation accuracy = {accuracies['valid']}")
  print(f"               diff = {accuracies['diff']}")

def make_confusion_matrix(model_name: str, model : RandomForestClassifier, X: pd.DataFrame, y: pd.DataFrame, outdir: str) -> None:
  """ Plots and saves a confusion matrix for the given model on the X & y data. """
  y_pred = model.predict(X)
  cm = confusion_matrix(y, y_pred)
  sns.heatmap(cm, annot=True, cmap=CMAP, xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"], fmt='g')
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.title(f"Confusion Matrix of Model of {model_name}")
  plt.savefig(os.path.join(outdir, f'confusion_matrix_{model_name}.png'))
  plt.close()

def evaluate_models(models: dict[str: RandomForestClassifier], X_train: pd.DataFrame, y_train: pd.DataFrame,
                    X_valid: pd.DataFrame, y_valid: pd.DataFrame, plots_dir: str) -> None:
  """
  Evaluates all of the models by computing their accuracy score in both the training and validation set.
  Then calculates the model with the greatest validation accuracy and the least overfit model and makes confusion
  matrices for each one. 
  """
  model_accuracies: dict[str:dict] = dict()
  for model_name, model in models.items():
    
    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    valid_accuracy = accuracy_score(y_valid, valid_pred)

    model_accuracies[model_name] = {'train' : train_accuracy, 'valid': valid_accuracy, 
                                    'diff': np.abs(valid_accuracy - train_accuracy)}
    
    print_accuracy_summary(model_name, model_accuracies[model_name])

  most_accurate_model = model_name # the most accurate model is that with the hightest validation accuracy.
  least_overfit_model = model_name # the least overfit model is that with the lowest diff.

  for model_name, accuracies in model_accuracies.items():
    if accuracies['valid'] > model_accuracies[most_accurate_model]['valid']:
      most_accurate_model = model_name
    if accuracies['diff'] < model_accuracies[least_overfit_model]['diff']:
      least_overfit_model = model_name

  print_accuracy_summary(f"Most Accurate Model ({most_accurate_model})", model_accuracies[most_accurate_model])
  print_accuracy_summary(f"Least Overfit Model ({least_overfit_model})", model_accuracies[least_overfit_model])  

  make_confusion_matrix(most_accurate_model, models[most_accurate_model], X_valid, y_valid, plots_dir)
  make_confusion_matrix(least_overfit_model, models[least_overfit_model], X_valid, y_valid, plots_dir)
          
def main() -> None:
  data = pd.read_csv('titanic_data.csv')
  data = preprocess_data(data)
  
  plots_dir = os.path.join(os.getcwd(), 'plots')
  if not os.path.isdir(plots_dir): os.mkdir(plots_dir)
  generate_plots(data.copy(), plots_dir)
  
  X_train, X_valid, y_train, y_valid = get_training_and_validation_data(data)
  models = train_models(X_train, y_train, MODEL_HYPERPARAMETERS_FOR_TESTING)
  evaluate_models(models, X_train, y_train, X_valid, y_valid, plots_dir)

if __name__ == "__main__":
  main()  