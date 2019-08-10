def plot_information_gain(model, num_features):
    '''
    This was pulled from stack overflow somewhere (and modified a bit)
    :param model: xgboost model
    :param num_features: Number of features to plot on feature importance plot( recommend <=50)
    :return: (void) creates plotly plot window
    '''

    import pandas as pd
    from matplotlib import pyplot as plt;plt.rcdefaults()
    importance_frame = model.get_score(importance_type='gain')
    importance_frame = pd.DataFrame({'Importance': list(importance_frame.values()), 'Feature': list(importance_frame.keys())})
    importance_frame.sort_values(by = 'Importance', inplace=True, ascending=[False])
    importance_frame = importance_frame.reset_index(drop=True)
    importance_frame.head(num_features).sort_values(by = 'Importance').plot(x ='Feature', y='Importance' ,kind='barh',legend=False)
    plt.show()



def plot_regress_corr(true_vals, pred_vals,x_range=None,title='Regression Correlation Plot', dot_size=4, samp_size = 20000,cmap ='RdYlBu_r'):
    '''
    :param true_vals: Actual target vals
    :param pred_vals: Model predicted target vals
    :param x_range(optional): tuple (LowerBound,UpperBound) specifying the x-axis bounds
    :param title(optional): String title for plot
    :param dot_size(optional): integer value for dot_size of scatter points
    :param samp_size(optional): size of random sample(plotting full dataset takes forever)
    :param cmap(optional): string specifying the colormap. look at matplotlib documentation for other options
    :return: Void
    '''
    import numpy as np
    from numpy.random import choice
    from matplotlib import pyplot as plt
    from scipy.stats import gaussian_kde
    from sklearn.metrics import mean_squared_error
    # check the lengths
    if len(true_vals) != len(pred_vals):
        print('The length of true_vals and pred_vals are not equal.')
        return

    if samp_size>len(true_vals):
        samp_size=len(true_vals)
    # remove unsuable values
    index = [i for i, j in enumerate(pred_vals) if np.isfinite(true_vals[i]) and np.isfinite(pred_vals[i])]
    true_vals = [true_vals[i] for i in index]
    pred_vals = [pred_vals[i] for i in index]

    # get correlation coef
    coef = str(round(np.corrcoef(true_vals, pred_vals)[1, 0], 3))
    r2 = str(round(np.square(np.float(coef)),3))
    legend_text = 'Correlation Coef: ' + coef + '\n' + 'R^2 Score: ' + r2 + '\n'

    if len(true_vals) > samp_size:
        index = choice(range(len(true_vals)), samp_size)
        true_vals = [true_vals[i] for i in index]
        pred_vals = [pred_vals[i] for i in index]

    # Calculate the point density
    try:
        xy = np.vstack([true_vals, pred_vals])
        z = gaussian_kde(xy)(xy)
    except:
        z = np.ones(len(true_vals))

    # calculate the axis max/mins
    if x_range is None:
        plt_min = np.nanpercentile(true_vals, 2)
        plt_max = np.nanpercentile(true_vals, 98)
    else:
        plt_min = min(x_range)
        plt_max = max(x_range)

    plt_range = plt_max - plt_min

    plt.scatter(true_vals, pred_vals, c=z, s=dot_size, cmap=cmap)
    plt.xlabel('True Values', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Values', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=16)
    plt.text(plt_min+plt_range*.03, plt_min+plt_range*.8, legend_text, fontsize=10,fontweight='bold')
    plt.ylim((plt_min, plt_max))
    plt.xlim((plt_min, plt_max))
    if plt_max > plt_min:
        plt.plot(range(int(np.floor(plt_min)), int(np.ceil(plt_max))+1, 1), range(int(np.floor(plt_min)), int(np.ceil(plt_max))+1, 1), 'k')



def plot_shap_univar(col, shap_values, features,logged_col=False, feature_names=None,cmap ='RdYlBu_r',xlim=None,ylim=None,
                     dynamic_axis = True, dot_size=16, alpha=1, title=None, show=True,plot_horiz=True):
    """
    Austin Mishoe
    Create a SHAP dependence plot, colored by an interaction feature.
    Parameters
    ----------
    col : Int or String associated with the column name
        Index of the feature to plot.
    shap_values : numpy.array
        Matrix of SHAP values (# samples x # features)
    features : numpy.array or pandas.DataFrame
        Matrix of feature values (# samples x # features)
    feature_names : list
        Names of the features (length # features)
    cmap : String/cmap value
        cmap to use in the plots
    xlim : tuple(x1,x2)
        x-limits to use in the plot
    """
    import matplotlib.pyplot as plt
    import gc
    import numpy as np
    # convert from DataFrame
    if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = features.columns
        features = features.as_matrix()

    def convert_name(col):
        if type(col) == str:
            nzinds = np.where(feature_names == col)[0]
            if len(nzinds) == 0:
                print("Could not find feature named: "+col)
                return None
            else:
                return nzinds[0]
        else:
            return col
    col = convert_name(col)

    # get both the raw and display feature values
    if logged_col==True:
        feature_vals = np.exp(features[:,col])
    else:
        feature_vals = features[:, col]
    shap_vals = shap_values[:,col]
    clow = np.nanpercentile(shap_values[:,col], 2)
    chigh = np.nanpercentile(shap_values[:, col], 98)
    if abs(clow)<abs(chigh):
        clow = -chigh
    else:
        chigh = -clow

    feature_name = feature_names[col]

    # the actual scatter plot
    plt.scatter(feature_vals, shap_vals, s=dot_size, linewidth=0, c=shap_vals,cmap=cmap,vmin=clow, vmax=chigh,
               alpha=alpha, rasterized=len(feature_vals) > 500,edgecolors='k', lw=.2)
    #plot colorbar
    plt.colorbar()
    plt.gcf().set_size_inches(6, 5)
    plt.xlabel(feature_name, fontsize=13)
    plt.ylabel("SHAP value for\n"+feature_name, fontsize=13)
    if plot_horiz==True:#plot horizontal line @ y=0
        plt.axhline(y=0.0, color='k', linestyle='-')
    if title != None:
        plt.title(title, fontsize=13)

    if dynamic_axis:
        LB = np.nanpercentile(feature_vals, .75)
        UB = np.nanpercentile(feature_vals, 99.25)
        plt.xlim((LB-.03*(UB-LB)), (UB+.03*(UB-LB)))

    if xlim != None:
        plt.xlim(xlim[0],xlim[1])
    if ylim != None:
        plt.ylim(ylim[0],ylim[1])
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().tick_params( labelsize=11)
    for spine in plt.gca().spines.values():
        spine.set_edgecolor("#333333")
    if show:
        plt.show()
    gc.collect()
