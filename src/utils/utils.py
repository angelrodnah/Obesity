import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def data_report(df):
    # Sacamos los NOMBRES
    cols = pd.DataFrame(df.columns.values, columns=["COL_N"])

    # Sacamos los TIPOS
    types = pd.DataFrame(df.dtypes.values, columns=["DATA_TYPE"])

    # Sacamos los MISSINGS
    percent_missing = round(df.isnull().sum() * 100 / len(df), 2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns=["MISSINGS (%)"])

    # Sacamos los VALORES UNICOS
    unicos = pd.DataFrame(df.nunique().values, columns=["UNIQUE_VALUES"])
    
    percent_cardin = round(unicos['UNIQUE_VALUES']*100/len(df), 2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns=["CARDIN (%)"])

    concatenado = pd.concat([cols, types, percent_missing_df, unicos, percent_cardin_df], axis=1, sort=False)
    concatenado.set_index('COL_N', drop=True, inplace=True)


    return concatenado.T


def drop_cols(df_, max_cardi=20, max_miss=30):
    df = df_.copy()
    delete_col = []
    
    for i in df.columns:
        missings = df[i].isnull().sum() * 100 / len(df)
        
        # Elimina por missings
        if missings >= max_miss:
            df.drop(i, 1, inplace=True)
            continue
        
        # Elimina por cardinalidad en variables categoricas
        if df[i].dtype.name in ('category', 'object'):
            if df[i].nunique()*100/len(df) >= max_cardi:
                df.drop(i, 1, inplace=True)          
        
    return df

from scipy.stats import iqr

def outliers_quantile(df, feature, param=1.5):  
        
    iqr_ = iqr(df[feature], nan_policy='omit')
    q1 = np.nanpercentile(df[feature], 25)
    q3 = np.nanpercentile(df[feature], 75)
    
    th1 = q1 - iqr_*param
    th2 = q3 + iqr_*param
    
    return df[(df[feature] >= th1) & (df[feature] <= th2)].reset_index(drop=True)

def outlier_meanSd(df, feature, param=3):   
    media = df[feature].mean()
    desEst = df[feature].std()
    
    th1 = media - desEst*param
    th2 = media + desEst*param

    return df[((df[feature] >= th1) & (df[feature] <= th2))  | (df[feature].isnull())].reset_index(drop=True)


def plot_tidy_categorical(df, sel_cols, target, file_output=None):
    """
    Generate bar plots for each categorical variable,
    grouped by target variable.

    Parameters:
    - df: DataFrame, the input dataset
    - sel_cols: list, categorical variables
    - target: str, name of the column containing the target variable

    Returns:
    - None (displays the plots)
    """
    # Filter columns to keep only selected categorical columns
    tidy_df = df[sel_cols]
    tidy_df[target] = df[target]

    # Melt DataFrame to long format
    tidy_df = tidy_df.melt(id_vars=target, value_vars=sel_cols, var_name='Var', value_name='Values')

    # Calculate percentages
    counts = tidy_df.groupby(['Var', 'Values', target]).size().reset_index(name='Counts')
    total_counts = counts.groupby(['Var', target])['Counts'].transform('sum')
    counts['Percentage'] = counts['Counts'] / total_counts * 100

    # Set appropriate font scale
    sns.set(font_scale=1)

    # Set up the facets with reduced aspect ratio and smaller height
    facets = sns.catplot(
        data=counts,
        x='Percentage',  # Rotating subplots to make y-axis the percentage
        y='Values',      # Rotating subplots to make x-axis the categorical values
        kind='bar',
        col='Var',
        hue=target,
        sharey=False,
        sharex=False,
        aspect=1.5,  # Reducing aspect ratio
        height=5,    # Reducing height
        col_wrap=3,
    )

    # Set title for each subplot
    facets.set_titles("{col_name}")

    # Rotate x-axis labels
    for ax in facets.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    if file_output:
        facets.savefig(file_output)


def plot_tidy(df, sel_cols, target,file_output=None):
    """
    Generate kernel density estimation (KDE) plots for each variable,
    grouped by target variable.
    
    Parameters:
    - df: DataFrame, the input dataset
    - sel_cols: list, variables used for clustering
    - target: str, name of the column containing the target variable
    
    Returns:
    - None (displays the plots)
    """
    # Index df on target variable
    tidy_df = df.set_index(target)
    # Keep only selected columns
    tidy_df = tidy_df[sel_cols]
    # Stack column names into a column, obtaining a "long" version of the dataset
    tidy_df = tidy_df.stack()
    # Take indices into proper columns
    tidy_df = tidy_df.reset_index()
    # Rename column names
    tidy_df = tidy_df.rename(columns={"level_1": "Var", 0: "Values"})
    

    sns.set(font_scale=1)
    
    facets = sns.FacetGrid(
        data=tidy_df,
        col="Var",
        hue=target,
        sharey=False,
        sharex=False,
        aspect=1.2,  
        height=4,    
        col_wrap=3,
    )
    # Build the plot from `sns.kdeplot` with smaller linewidth
    _ = facets.map(sns.kdeplot, "Values", shade=True, linewidth=1.5).add_legend()
    # Set titles with smaller font size
    facets.set_titles("{col_name}", fontsize=12)
    # Set x-axis label for all facets
    facets.set_ylabels("Densidad")
    facets.set_xlabels("Valores")
    
    if file_output:
        facets.savefig(file_output)
        
def reduce_memory_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    # Excluir las columnas de índice
    columns_to_exclude = df.index.names if df.index.name else []
    for col in df.columns:
        if col in columns_to_exclude:
            continue
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

from scipy.stats import kurtosis, skew
import statsmodels.api as sm
from scipy.stats import norm


def plot_kde_histogram_with_stats(data, column_name, title, ax=None):
    # Use provided ax or create a new subplot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # KDE + Histogram
    sns.distplot(data[column_name],fit=norm, kde=True, rug=True, ax=ax)
    ax.set_title(f'KDE + Histogram - {title}')

    # Verificar si la columna contiene datos numéricos
    if pd.api.types.is_numeric_dtype(data[column_name].dtype):
        # Calcular estadísticas solo si la columna contiene datos numéricos
        median_val = np.median(data[column_name])
        mean_val = np.mean(data[column_name])
        mode_val = data[column_name].mode().values[0]
        variance_val = np.var(data[column_name])
        devstd_val = np.std(data[column_name])
        kurtosis_val = kurtosis(data[column_name])
        skewness_val = skew(data[column_name])
    else:
        # Si la columna no es numérica, asignar valores predeterminados
        variance_val = devstd_val = kurtosis_val = skewness_val = np.nan
    
    
    kurtosis_val = kurtosis(data[column_name])
    skewness_val = skew(data[column_name])

    # Add vertical lines for median, mean, and mode
    ax.axvline(median_val, color='red', linestyle='dashed', linewidth=2)
    ax.axvline(mean_val, color='green', linestyle='dashed', linewidth=2)
    ax.axvline(mode_val, color='blue', linestyle='dashed', linewidth=2)

    # Height of the plot
    height = ax.get_ylim()[1]

    # Add text labels
    ax.text(median_val, height / 2, f'Mediana: {median_val:.2f}', color='red', rotation=90, verticalalignment='center', horizontalalignment='right')
    ax.text(mean_val, height / 2, f'Media: {mean_val:.2f}', color='green', rotation=90, verticalalignment='center', horizontalalignment='right')
    ax.text(mode_val, height / 2, f'Moda: {mode_val:.2f}', color='blue', rotation=90, verticalalignment='center', horizontalalignment='right')

    # Add text with statistics
    text = f'Varianza: {variance_val:.2f}\nDesviación Std.: {devstd_val:.2f}\nCurtosis: {kurtosis_val:.2f}\nSimetría: {skewness_val:.2f}'
    ax.text(0.95, 0.9, text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', edgecolor='grey', facecolor='white'))

    # Set y-label
    ax.set_ylabel("Frecuencia")
    
    # Añadir texto en la parte inferior de la figura
    sim,curtosis=sim_curtosis(data, column_name)
    plt.figtext(0.05, -0.05, f'{sim}\n{curtosis}', ha="left", fontsize=8, bbox={"facecolor":"white", "alpha":0.7, "pad":15})
    
    return ax

def box_violin_plot(data, column, title, ax=None):
    # Use provided ax or create a new subplot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # Violin + Boxplot
    sns.violinplot(data=data, x=column, inner='point', linewidth=0, saturation=0.4, orient='h', ax=ax)
    sns.boxplot(x=column, data=data, width=0.3, boxprops={'zorder': 2}, ax=ax, orient='h', fliersize=5)

    # Adjust the plot design
    ax.set_title(f"Boxplot + Violin Plot - {title}")

    # Get statistics
    median = data[column].median()
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    threshold_up = q3 + 1.5 * iqr

    # Insert text into the plot
    ax.text(median, 0.35, f'Mediana: {median:.2f}', ha='center', va='center', color='red', fontsize=10, rotation=90)
    ax.text(0.97 * q1, 0.35, f'Q1: {q1:.2f}', ha='center', va='center', color='blue', fontsize=10, rotation=90)
    ax.text(0.97 * q3, 0.35, f'Q3: {q3:.2f}', ha='center', va='center', color='blue', fontsize=10, rotation=90)
    ax.text(0.97 * threshold_up, 0.25, f'Outliers: {threshold_up:.2f}', ha='center', va='center', color='blue', fontsize=10, rotation=90)

    for elem in [q1, q3, threshold_up]:
        ax.axvline(elem, color='black', linestyle='--', label=f'{elem:.2f}')

    return ax

def qq_plot(data, column_name, title,ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    
    # Create QQ plot
    sm.qqplot(data[column_name], line='s', ax=ax)
    ax.set_title(f"Quantile-Quantile Plot - {title}")
    plt.xlabel("Cuantiles teóricos")
    plt.ylabel("Cuantiles observados")

def plot_distribucion(data, column_name, **kwargs):
    title = kwargs.get('title', column_name)
    save_png, filename = kwargs.get('save_png', False), kwargs.get('filename', 'tmp.png')

    # Create subplots with specified width ratios
    fig, axs = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 2, 1]})

    # Loop through the plots and functions
    for i, plot_function in enumerate([plot_kde_histogram_with_stats, box_violin_plot, qq_plot]):
        plot_function(data, column_name, title, ax=axs[i])

    # Adjust the layout of the subplots
    plt.tight_layout()
    plt.show()

    # Call additional function
    sim_curtosis(data, column_name)

    # Check and save the figure
    if save_png:
        fig.savefig(filename)

def sim_curtosis(data, column_name):
    kurtosis_valor = kurtosis(data[column_name])
    skewness_valor = skew(data[column_name])

    if kurtosis_valor > 3:
        kurtosis_="La distribución es leptocúrtica, lo que sugiere colas pesadas y picos agudos."
    elif kurtosis_valor < 3:
        kurtosis_="La distribución es platicúrtica, lo que sugiere colas ligeras y un pico achatado."
    else:
        kurtosis_="La distribución es mesocúrtica, similar a una distribución normal."

    if skewness_valor > 0:
        simetria_="La distribución es asimétrica positiva (sesgo hacia la derecha)."
    elif skewness_valor < 0:
        simetria_="La distribución es asimétrica negativa (sesgo hacia la izquierda)."
    else:
        simetria_="La distribución es perfectamente simétrica alrededor de su media."
    return simetria_,kurtosis_

def plot_horizontal_catplot(df, catcols, diccionario_columnas=None, diccionario_valores=None,
                            save_png=False, filename='tmp.png'):
    """
    Genera gráficos Seaborn para la frecuencia de valores únicos en las columnas categóricas del DataFrame.
    Los gráficos se organizan en filas y las categorías se ordenan por porcentaje descendente.
    El número de valores NaN o nulos se muestra en el título de cada gráfico.

    Parameters:
    - df: DataFrame de pandas
    - catcols: Lista de columnas categóricas
    - diccionario_columnas: Diccionario para decodificar nombres de columnas (opcional)
    - diccionario_valores: Diccionario de diccionarios para decodificar valores (opcional)

    Returns:
    - No devuelve nada, pero muestra gráficos para cada columna categórica.
    """
    for column in catcols:
        plt.figure(figsize=(10, 5))

        # Decodificar nombre de la columna si se proporciona un diccionario
        clabel = diccionario_columnas.get(column, column) if diccionario_columnas else column

        # Decodificar valores si se proporciona un diccionario y existe un diccionario para la columna
        decoded_df = df.copy()
        if diccionario_valores and column in diccionario_valores:
            decoded_df[column] = df[column].map(diccionario_valores[column])

        # Convertir variables numéricas a tipo "object"
        if pd.api.types.is_numeric_dtype(decoded_df[column].dtype):
            decoded_df[column] = decoded_df[column].astype('object')

        # Calcular porcentaje de frecuencia y ordenar por porcentaje descendente
        percentages = decoded_df[column].value_counts(normalize=True) * 100
        nan_count = df[column].isnull().sum()

        if percentages.empty:
            print(f"No hay datos para graficar en la columna {column}.")
            continue

        order = percentages.index
        ax = sns.barplot(x=percentages, y=order, palette="viridis", order=order)

        plt.ylabel(clabel)
        plt.title(f'Distribución de valores {clabel} ( {column} )\nNulos: {nan_count}', fontsize=10)
        plt.xlabel('Frecuencia (%)')

        # Anotaciones a la derecha de las barras
        for index, value in enumerate(percentages):
            plt.text(value + 0.5, index, f'{int(value)}%', fontsize=8, va='center')

        plt.xticks(rotation=90)
        # Añadir el grid en el eje x con un paso del 5%
        x_ticks = range(0, 105, 5)
        plt.xticks(x_ticks)

        if save_png:
            plt.savefig(filename)
        plt.show()

def plot_analysis(data, target, col):
    col_mean = []

    for each in df[target].unique():
        x = data[data[target] == each]
        mean = x[col].mean()
        col_mean.append(mean)

    plt.figure(figsize=(8,6))
    plt.subplot(2,2,1)
    plt.hist(data[col], color="lightgreen")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.title(f"{col} histogram", color="black", fontweight='bold', fontsize=6)
    
    plt.subplot(2,2,2)
    sns.distplot(data[col], fit=norm, color="green")
    plt.title(f"{col} Distplot", color="black", fontweight='bold', fontsize=6)
    
    plt.subplot(2,2,3)
    sns.barplot(x=df[target].unique(), y=col_mean, palette="Greens")
    plt.title(f"The average value of {col} by {target}", color="black", fontweight='bold', fontsize=6)
    plt.xlabel(target)
    plt.ylabel(f"{col} mean")
    
    plt.subplot(2,2,4)
    sns.boxplot(x=data[target], y=data[col], palette='Greens') 
    plt.title(f"{col} & {target}", color="black", fontweight='bold', fontsize=6)
    
    plt.tight_layout()
    plt.show()

# Funciones para la transformación de las características numéricas
def sqrt_transform(X):
    return np.sqrt(X)

def log_transform(X):
    return np.log1p(X)
def classify_distributions(df, threshold=0.05):
    """
    Clasifica las distribuciones de las columnas numéricas del DataFrame en una de las siguientes categorías:
    - 'normal': si la distribución se ajusta a una distribución normal según el test de Shapiro-Wilk.
    - 'positive_increasing': si la distribución es estrictamente creciente positiva.
    - 'positive_decreasing': si la distribución es estrictamente decreciente positiva.
    - 'rectangular': si la distribución es rectangular.
    - 'skewed_left': si la distribución tiene sesgo a la izquierda.
    - 'skewed_right': si la distribución tiene sesgo a la derecha.
    - 'bimodal': si la distribución tiene dos modas.
    - 'trimodal': si la distribución tiene tres modas.
    - 'polimodal': si la distribución tiene más de tres modas.
    - 'other': si no encaja en ninguna de las categorías anteriores.
    
    Parameters:
    - df: DataFrame. El DataFrame que contiene las características.
    - threshold: float. Umbral para clasificar la distribución como normal según el test de Shapiro-Wilk.

    Returns:
    - dist_class: dict. Un diccionario que mapea el nombre de cada columna con una tupla que contiene
                  el tipo de distribución y el nombre de la transformación propuesta.
    """
    # Seleccionar solo las columnas numéricas
    numeric_cols = df.select_dtypes(include=['number']).columns

    dist_class = {}

    for col in numeric_cols:
        data = df[col].dropna()
        _, p_value = shapiro(data)

        # Comprobar si todos los valores son positivos
        all_positive = all(x > 0 for x in data)
        
        if p_value > threshold:
            dist_class[col] = ('normal', None)
        else:
            # Calcular sesgo y curtosis
            skewness = skew(data)
            kurt = kurtosis(data)

            # Clasificar distribuciones
            if skewness > 0 and kurt > 0 and all_positive:
                dist_class[col] = ('positive_increasing', 'sqrt')
            elif skewness < 0 and kurt > 0 and all_positive:
                dist_class[col] = ('positive_decreasing', 'log')
            elif skewness > 0 and kurt < 0 and all_positive:
                dist_class[col] = ('skewed_right', 'yeo-johnson')
            elif skewness < 0 and kurt < 0 and all_positive:
                dist_class[col] = ('skewed_left', 'yeo-johnson')
            elif skewness == 0 and kurt == 0:
                dist_class[col] = ('rectangular', None)
            elif skewness == 0 and kurt > 0:
                dist_class[col] = ('rectangular', None)
            elif skewness == 0 and kurt < 0:
                dist_class[col] = ('rectangular', None)
            else:
                # Clasificar distribuciones multimodales
                modes = len(pd.Series(data).mode())
                if modes == 2:
                    dist_class[col] = ('bimodal', 'yeo-johnson')
                elif modes == 3:
                    dist_class[col] = ('trimodal', 'yeo-johnson')
                elif modes > 3:
                    dist_class[col] = ('polimodal', 'yeo-johnson')
                else:
                    dist_class[col] = ('other', None)

    return dist_class

def selvars_boruta(df,target):
    
    Feature_Selector = BorutaShap(importance_measure='shap',
                                classification=True)
    Feature_Selector.fit(X=df, y=df[target], n_trials=100, random_state=0)
    Feature_Selector.TentativeRoughFix()
    Feature_Selector.plot(X_size=8, figsize=(20,8),
                y_scale='log', which_features='all')
    selvars=sorted(list(Feature_Selector.Subset().columns))
    
    return selvars
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(true_labels, predicted_labels):
    """
    Plot a confusion matrix using true labels and predicted labels.

    Parameters:
    true_labels (array-like): True labels.
    predicted_labels (array-like): Predicted labels.
    """
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')

    # Add labels and title
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Show the plot
    plt.show()

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def cross_validation_with_confusion_matrix(estimator, X, y, nsplits=10):
    cv = StratifiedKFold(n_splits=nsplits)
    # Perform cross-validation
    predicted = cross_val_predict(estimator, X, y, cv=cv)

    # Calculate confusion matrix and classification report for each fold
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fit the estimator on training data
        estimator.fit(X_train, y_train)

        # Predict on test data
        y_pred = estimator.predict(X_test)

        # Calculate confusion matrix and classification report
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))

        # Print confusion matrix and classification report
        print(f"Fold {i+1}:")
        print("Confusion Matrix:")
        print(conf_matrix)
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix - Fold {i+1}')
        plt.show()
        print("\nClassification Report:")
        return(class_report)
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def plot_roc_curve(model, X_val, y_val):
    # Predecir las probabilidades de las clases positivas
    y_prob = model.predict_proba(X_val)[:, 1]
    
    # Calcular la tasa de verdaderos positivos (TPR) y la tasa de falsos positivos (FPR)
    fpr, tpr, thresholds = roc_curve(y_val, y_prob)
    
    # Calcular el área bajo la curva ROC (AUC)
    auc = roc_auc_score(y_val, y_prob)
    
    # Plotear la curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

# Llama a la función con tu modelo y datos de validación
#plot_roc_curve(model_tuned, X_val, y_val)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def generate_roc_auc(estimator, X_train, y_train, X_val, y_val):
    """
    Generate ROC curve and calculate AUC using one-vs-all technique.

    Parameters:
    estimator: scikit-learn estimator object
        The classifier or regressor to use.
    X_train: array-like, shape (n_samples, n_features)
        The input samples for training.
    y_train: array-like, shape (n_samples,)
        The target values for training.
    X_val: array-like, shape (n_samples, n_features)
        The input samples for validation.
    y_val: array-like, shape (n_samples,)
        The target values for validation.

    Returns:
    None
    """
    # Convert the multiclass problem into binary using one-vs-all technique
    clf = OneVsRestClassifier(estimator)

    # Train the binary classifier using training data
    clf.fit(X_train, y_train)

    # Calculate prediction probabilities for each class in the binary classifier
    y_score = clf.predict_proba(X_val)

    # Convert y_val to a pandas Series and select only the first column
    y_val_series = pd.Series(y_val.iloc[:, 0])

    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(clf.classes_)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_val_series == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve (class {}) - AUC = {:.2f}'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
def hyperparameter_tuning(models, X, y):
    # Almacena los resultados en un DataFrame
    results = []

    # Loop sobre los modelos
    for model_name, config in models.items():
        print(f"Tuning hyperparameters for {model_name}")
        model = RandomizedSearchCV(config["model"], config["params"], n_iter=50, cv=5, verbose=2, random_state=42)
        model.fit(X, y)  # Ajusta el modelo con los datos de entrenamiento
        best_params = model.best_params_
        best_score = model.best_score_
        results.append([model_name, best_params, best_score])

    # Crear un DataFrame a partir de los resultados
    results_df = pd.DataFrame(results, columns=["Model", "Best Parameters", "Best Score"])

    return results_df

from sklearn.model_selection import StratifiedKFold,cross_val_score

def perform_cross_validation(models, x_train, y_train, n_splits=8, random_state=42,metric='accuracy'):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    cv_results = []
    for name, model in models.items():
        cv_results.append(cross_val_score(model, x_train, y_train, scoring=metric, cv=kfold, n_jobs=-1))
    
    cv_means = [cv_result.mean() for cv_result in cv_results]
    cv_std = [cv_result.std() for cv_result in cv_results]
    
    cv_df = pd.DataFrame({"CrossVal_Score_Means": cv_means, "CrossValerrors": cv_std, "Algorithm": list(models.keys())})
    
    plt.figure(figsize=(10, 7))
    g = sns.barplot(x="CrossVal_Score_Means", y="Algorithm", data=cv_df, orient="h", palette='cool', 
                    edgecolor="black", linewidth=1)
    g.set_xlabel("Mean "+ metric, fontsize=18)
    g.set_title("Cross validation scores", fontsize=24)
    plt.show()
    
    return cv_df