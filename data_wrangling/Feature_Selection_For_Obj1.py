import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler

def fit_standard_scaler(df):
    """
    Applies standard scaling to a DataFrame.

    Parameters:
    df : A DataFrame containing the numerical features to be scaled.

    Returns:
    scaled_data : A numpy array where each feature in the input DataFrame 
    has been scaled to have a mean of zero and a standard deviation of one.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data

def fit_min_max_scaler(df):
    """
    Applies min-max scaling to a DataFrame.

    Parameters:
    df : A DataFrame containing the numerical features to be scaled.

    Returns:
    scaled_data : A numpy array where each feature in the input DataFrame 
    has been scaled to have a mean of zero and a standard deviation of one.
    """
    scaler = MinMaxScaler()
    min_max_scaled_data = scaler.fit_transform(df)
    return min_max_scaled_data

def PCA_explained_variance_ratio(df):
    """
    Computes the explained variance ratio of each principal component.

    This function initializes and fits a PCA model to the data provided in the DataFrame.
    It calculates the ratio of variance that each principal component accounts for in the dataset.

    Parameters:
    df : A DataFrame containing the data on which to perform PCA. All columns should be numerical.

    Returns:
    explained_variance_ratio : 
    An array containing the proportion of variance explained by each of the selected components.
    """
    pca = PCA()
    pca.fit(df)

    explained_variance_ratio = pca.explained_variance_ratio_
    return explained_variance_ratio

def PCA_principal_component(df, n_components):
    """
    Performs Principal Component Analysis (PCA) on the provided dataset.

    Parameters:
    df : A DataFrame containing the data to perform PCA on.
    n_components : The number of principal components to calculate.

    Returns:
    pca_2d : The fitted PCA model.
    principal_components : The array of transformed data in the new principal component space.
    """
    # Transform the scaled data using the first two principal components
    pca_2d = PCA(n_components = n_components)
    principal_components = pca_2d.fit_transform(df)
    return pca_2d, principal_components

def NMF_principal_component(df, n_components):
    """
    Performs NMF on the provided dataset.

    Parameters:
    df : A DataFrame containing the data to perform NMF on.
    n_components : The number of components to calculate.

    Returns:
    W : The transformed data matrix with non-negative elements, representing the coefficients in the NMF model.
    H : The matrix containing the feature loadings, representing the contribution of each feature to the components.
    """
    nmf_2d = NMF(n_components=n_components, init='random', random_state=0)

    W = nmf_2d.fit_transform(df)
    # Extract the loadings (V) of the original features on the principal components
    H = nmf_2d.components_.T
    return W, H


def plot_heatmap(df, xlabel, ylabel, fig_name, show = True, save = False):
    """
    Creates a heatmap visualization of the provided DataFrame.

    Parameters:
    df : The DataFrame containing the data to be visualized as a heatmap.
    xlabel : The label for the x-axis, which describes the columns of the DataFrame.
    ylabel : The label for the y-axis, which describes the rows of the DataFrame.
    fig_name : The name of the file to save the figure to if saving is enabled.
    show : A boolean variable showing whether to display the plot on the screen. Defaults to True.
    save : A boolean variable showing Whether to save the plot to a file. Defaults to False.

    Returns:
    None
    """
    # Plotting the heatmap
    sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".2f")
    #plt.title('Heatmap of Feature Loadings on Principal Components')
    plt.ylabel(ylabel=ylabel, fontsize=15)
    plt.xlabel(xlabel = xlabel, fontsize=15)
    plt.tight_layout()

    show_save(show = show, save = save, fig_name=fig_name)

def scatterplot(feature_1, feature_2, xlabel, ylabel, fig_name, c = None, alpha = 1, need_legend = False, 
                legend_title = None, show = False, save = False):
    """
    Generates a scatter plot with the provided features and options.

    Parameters:
    feature_1 : The data for the x-axis of the scatter plot.
    feature_2 : The data for the y-axis of the scatter plot.
    xlabel : The label for the x-axis.
    ylabel : The label for the y-axis.
    fig_name : The filename for saving the figure.
    c : The color or sequence of colors for the markers.
    alpha : The alpha blending value, between 0 (transparent) and 1 (opaque).
    need_legend : If True, display a legend using the labels detected in the 'c' parameter.
    legend_title : The title for the legend.
    show : A boolean variable showing whether to display the plot on the screen. Defaults to True.
    save : A boolean variable showing Whether to save the plot to a file. Defaults to False.

    Returns:
    None
    """
    fig = plt.gcf()  # Get the current figure
    scatter = plt.scatter(feature_1, feature_2, alpha=alpha, c = c)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    if need_legend:
        plt.legend(*scatter.legend_elements(), title = legend_title,fontsize=10)
    plt.grid(False)
    plt.tight_layout()

    show_save(show = show, save = save, fig_name = fig_name)

def show_save(show, save, fig_name):
    """
    This function will save the current plot to the 'data_visualization' directory with the provided filename if 'save' is True. 

    Parameters:
    show: A boolean indicating whether to display the plot.
    save: A boolean indicating whether to save the plot to a file.
    fig_name: The filename for the plot if the save option is True.

    Returns:
    None
    """
    if save:
        plt.savefig('./data_wrangling/' + fig_name)
    
    if show:
        plt.show()
        return
    
    plt.cla()
    plt.clf()



if __name__ == '__main__':

    data = pd.read_csv('./data/monthly_data_16_23.csv')

    """
    PCA
    """
    # Remove YearMonth since it is not a feature
    numerical_data = data.drop(['YearMonth'], axis=1)

    # Remove Call Count since it is our label
    numerical_data = numerical_data.drop(['Call Count'], axis=1)
    
    # Standardization
    scaled_data = fit_standard_scaler(numerical_data)

    explained_variance_ratio = PCA_explained_variance_ratio(scaled_data)
    # Apply n_components = 2 on PCA
    pca_2d, principal_components = PCA_principal_component(scaled_data, 2)
    
    # Plot the transformed data
    scatterplot(principal_components[:, 0], principal_components[:, 1], alpha = 1, c = data['Call Count'], xlabel='Principal Component 1', ylabel='Principal Component 2', 
                need_legend= True, legend_title="Call Volume", fig_name='scatterplot of 2016-2023 dataset in PC values', show = False, save = True)

    # Extract the loadings (V) of the original features on the principal components
    loadings = pca_2d.components_.T
    loadings = [np.abs(arr) for arr in loadings]
    # Create a dataframe for easier plotting
    features = numerical_data.columns
    loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=features)

    plot_heatmap(loadings_df, xlabel='Feature', ylabel='Principal Component', fig_name='Heatmap of Feature Loadings on Principal Components', show = False, save = True)

    cumulative_variance_explained = np.cumsum(explained_variance_ratio)

    # Creating a dataframe to display the percentage of variance explained by each principal component
    # and the cumulative percentage of variance explained
    pve_df = pd.DataFrame({
        'Principal Component': ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'],
        'Percentage of Variance Explained': explained_variance_ratio * 100,
        'Cumulative PVE': cumulative_variance_explained * 100
    })

    print(pve_df)

    # Plotting both the individual and cumulative percentage of variance explained
    plt.plot(pve_df['Principal Component'], pve_df['Percentage of Variance Explained'], marker='o', label='Individual PVE')
    plt.plot(pve_df['Principal Component'], pve_df['Cumulative PVE'], marker='x', linestyle='--', label='Cumulative PVE')

    #plt.title('Percentage of Variance Explained by PCA Components')
    plt.xlabel('Principal Components', fontsize=15)
    plt.ylabel('Percentage of Variance Explained', fontsize=15)
    plt.legend(loc='best',fontsize=15)
    plt.tight_layout()

    show_save(show = False, save = True, fig_name='Percentage of Var Explained')

    """
    NMF
    """
    min_max_scaled_data = fit_min_max_scaler(numerical_data)

    W, H = NMF_principal_component(min_max_scaled_data, 2)

    scatterplot(W[:, 0], W[:, 1], xlabel='Component 1', ylabel='Component 2', fig_name='scatter plot of 16-23 dataset in NMF values', c=data['Call Count'], need_legend=True, legend_title = 'Call Volume', show = False, save = True)

    # Create a dataframe for easier plotting
    features = numerical_data.columns
    loadings_df = pd.DataFrame(H, columns=['C1', 'C2'], index=features)

    plot_heatmap(loadings_df, xlabel='Feature', ylabel='Components', fig_name= 'Heatmap of Feature Loadings on NMF components', show = False, save = True)
