#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 18:26:48 2024

@author: Khadiatou Ly
NUID: 002786336
DS2500 programming with Data
Analysis of the wealth disparity
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

FILE = "state_wealth_inequality.csv"

def read_file(file):
    '''Read and clean the CSV file'''
    data = pd.read_csv(file)
    data = data.drop_duplicates()
    # dropping empty rows and ensuring cols are numeric
    data = data.dropna(subset=["state", "own", "college", "wealth_mean"])
    numeric_columns = ["wealth_mean", "own", "college"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data

def analyze_indicators(data, indicators):   
    '''Preliminary analysis of key indicators and their relationships to 
        wealth inequality'''
    # statistical analysis and correlation
    stats =  data[indicators].describe()
    corr = data[indicators].corr()
    print("========== Indicator Analysis ==========\n")
    print(stats)
    print("\n---------------------------------------\n")
    print(f"Correlation between {','.join(indicators)} ")
    print(corr,round(2))

def regression_analysis(data, x, y):
    '''Perform linear regression and return slope, intercept, and R²'''
    model = LinearRegression()
    model.fit(data[[x]], data[y])
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(data[[x]], data[y])
    print(f"Regression Analysis for {x} vs {y}:\n"
          f" y = {intercept:.2f} + {slope:.2f}x, R² = {r2:.2f}")
    return {"slope": slope, "intercept": intercept, "r2": r2}

def classification_model(data):
    '''Classify states as high or low wealth inequality'''
    data['high_inequality'] = (data['wealth_mean'] \
                               > data['wealth_mean'].median()).astype(int)
    # features and target
    features = ['college', 'own', 'wealth_mean']
    target = 'high_inequality'
    # training and testing sets
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,\
                                                        random_state=42)
    # standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # modeling
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    # Evaluate the model
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy:.2f}")


def clustering_model(data):
    '''Cluster states based on wealth-related features including homeownership'''
    # feautres
    features = ['college', 'own', 'wealth_mean']
    X = StandardScaler().fit_transform(data[features])  # Standardize the features

    #  KMeans clustering with 3 clusters 
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['cluster'] = kmeans.fit_predict(X)

    # clusters labels
    cluster_labels = {
        0: "High Wealth, High Education, High Homeownership",
        1: "Medium Wealth, Medium Education, Medium Homeownership",
        2: "Low Wealth, Low Education, Low Homeownership"
    }
    data['cluster_label'] = data['cluster'].map(cluster_labels)

    # cluster center findings
    print("\n=== Cluster Centers ===")
    for i, center in enumerate(kmeans.cluster_centers_):
        print(f"Cluster {i+1} Center: {center}")
    # visualize the clustering results
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x='college', y='wealth_mean', hue='cluster_label', palette='Set2')
    plt.title("Clustering of States by Wealth, Homeownership, and Education Features")
    plt.xlabel("College Education Rate (%)")
    plt.ylabel("Wealth Mean")
    plt.legend(title="Cluster Characteristics")
    plt.tight_layout()
    plt.show()
    return kmeans.cluster_centers_


def plot_and_analyze(data, x, y, x_label, y_label, title, color):
    '''Visualize relationships between key indicators and wealth inequality'''
    # linear regression parameters (slope, intercept, r2)
    regression_results = regression_analysis(data, x, y)
    slope, intercept, r2 = regression_results.values()
    x_vals = np.linspace(data[x].min(), data[x].max(), 100).reshape(-1, 1)
    y_preds = intercept + slope * x_vals
    
    # plot
    plt.scatter(data[x], data[y], alpha=0.6, color=color)
    plt.plot(x_vals, y_preds, color=color, linestyle='--', \
                                     label="Regression Line")
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return f"{title} Regression Line:\
        y = {intercept:.2f} + {slope:.2f}x\nR²: {r2:.2f}"

def racial_data(data):
    '''Reshape the data to include race and corresponding wealth mean values'''
    racial_columns = ['white', 'black', 'hispanic']
    melted_data = pd.melt(data, id_vars=["state", "year", "wealth_mean",\
    "own", "college"],value_vars=racial_columns, var_name="race", \
    value_name="race_wealth_mean")
    return melted_data

def top_bottom_states(data, n=10):
    '''Get top and bottom 10 states based on wealth mean'''
    sorted_states = data.sort_values("wealth_mean", ascending=False)
    top_states = sorted_states.head(n)["state"]
    bottom_states = sorted_states.tail(n)["state"]
    return top_states, bottom_states

def plot_states_and_races(data, top_states, bottom_states):
    """Plot wealth inequality by race for top and bottom states"""
    # filtering and extracting bottom and top states for plot
    filtered_data = data[data["state"].isin(top_states + bottom_states)]
    state_order = list(top_states) + list(bottom_states)
    print(state_order)
    # plot for top 10 bottom and top states 
    plt.figure(figsize=(14, 8))
    sns.barplot(data=filtered_data, x="state", y="race_wealth_mean", \
                hue="race", palette="Set2", ci=None, order=state_order)
    plt.title("Wealth Inequality by Race (Top/Bottom 10 States)", fontsize=16)
    plt.xlabel("State", fontsize=12)
    plt.ylabel("Mean Wealth by Race", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(title="Race")
    plt.tight_layout()
    plt.show()


def extract_states(data):
    '''Group by state and calculate mean wealth, homeownership, 
    and college education rates'''
    state_summary = data.groupby("state")[["wealth_mean", "own", \
                                           "college"]].mean().reset_index()
    return state_summary

def plot_by_state(state_summary):
    '''Visualizes mean wealth by state using a non-interactive plot'''
    plt.figure(figsize=(12, 8))  # Increase figure size for better spacing
    bars = plt.bar(state_summary['state'], state_summary['wealth_mean'], 
                   color=plt.cm.viridis(state_summary['wealth_mean'] / 
                                        max(state_summary['wealth_mean'])))
    
    plt.title("Mean Wealth by State", fontsize=16)
    plt.xlabel('State', fontsize=12)
    plt.ylabel('Mean Wealth', fontsize=12)
    plt.xticks(rotation=75, ha='right', fontsize=8)
    plt.tight_layout()
    plt.show()

   
def analyze_group_by(data, group_col, target_col):
    '''Analyze target metrics grouped by a categorical column'''
    print(f"========== Analysis Grouped by {group_col} ==========")
    summary = data.groupby(group_col)[target_col].describe()
    print(summary)
    return summary
    
    
def plot_group_by(data, group_col, target_col, title, xlabel, ylabel):
    '''Plot target metrics grouped by a categorical column'''
    summary = data.groupby(group_col)\
              [target_col].agg(['mean', 'std']).reset_index()
    plt.figure(figsize=(8, 6))
    sns.barplot(data=summary, x=group_col, y='mean', palette='Set2', \
                                                            capsize=0.2)
    plt.errorbar(x=summary[group_col], y=summary['mean'], yerr=summary['std'],\
                             fmt="none", ecolor="black", capsize=5, capthick=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def wealth_2020(data):
    '''Analyze and visualize wealth inequality for 2020'''
    data_2020 = data[data["year"] == 2020]
    if not data_2020.empty:
        features = data_2020[["college", "own", "wealth_mean"]]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        corr_matrix = pd.DataFrame(scaled_features, columns=["college", \
                                                "own", "wealth_mean"]).corr()
        print("\n=== Correlation Matrix (2020) ===")
        print(corr_matrix)
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", 
                                                        linewidths=0.5)
        plt.title("Correlation Heatmap (2020) - Scaled Data", fontsize=14)
        plt.tight_layout()
        plt.show()


def run_indicator_analysis(data, indicators):
    '''Run the analysis and plot for indicators'''
    analyze_indicators(data, indicators)
    for indicator, color in {"own": "purple", "college": "blue"}.items():
        plot_and_analyze(data, indicator, "wealth_mean", \
        f"{indicator.title()} Rate (%)", "Wealth Mean", \
        f"{indicator.title()} vs Wealth Inequality", color)

def main():
    # Load data
    disparity = read_file(FILE)
    race_data = racial_data(disparity)
    indicators = ["wealth_mean", "own", "college"]
    
    # ============= Indicator Analysis =============
    run_indicator_analysis(disparity, indicators)
    
    # ============= Race-Based Analysis =============
    analyze_group_by(race_data, "race", "race_wealth_mean")
    plot_group_by(race_data, "race", "race_wealth_mean", \
    "Mean Wealth Inequality by Race", "Race", "Mean Wealth")
   
    # ============= State-Based Analysis =============
    state_summary = extract_states(disparity)
    plot_by_state(state_summary)
   
    # Analyze top and bottom 10 states based on wealth mean
    top_states, bottom_states = top_bottom_states(state_summary)
    plot_states_and_races(race_data,top_states.tolist(),bottom_states.tolist()) 
   
    # ============= Year 2020 Analysis =============
    wealth_2020(disparity)
    
    # ============= Machine Learning Models =============
    classification_model(disparity)
    cluster_centers = clustering_model(disparity)
    print("Cluster Centers:\n", cluster_centers)
if __name__ == "__main__":
    main()

