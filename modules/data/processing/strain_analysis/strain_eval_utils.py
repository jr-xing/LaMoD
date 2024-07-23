import numpy as np
def abs_error(preds, GTs):
    errors = np.mean(np.abs(preds - GTs))
    return errors

def intraclass_correlation_coefficient(preds, GTs):
    """
    Calculate the intraclass correlation coefficient (ICC) between predicted and ground truth values.

    Args:
        preds (list or ndarray): List of predicted values.
        GTs (list or ndarray): List of ground truth values.

    Returns:
        float: Intraclass correlation
    """
    assert len(preds) == len(GTs), "Predicted and ground truth values must have the same length."
    n = len(preds)
    # mean_pred = np.mean(preds)
    # mean_GT = np.mean(GTs)
    x_bar = np.sum(preds+GTs) / (2 * n)
    s2 = np.sum((preds - x_bar)**2 + (GTs - x_bar)**2) / (2 * n)
    r = np.sum((preds - x_bar) * (GTs - x_bar)) / (n * s2)
    return r
    # SSB = n * (mean_pred - mean_GT)**2
    # SSW = np.sum((preds - mean_pred)**2) + np.sum((GTs - mean_GT)**2)
    # ICC = (SSB / (SSB + SSW))
    # return ICC

def intraclass_correlation_coefficient_copliot(preds, GTs):
    """
    Calculate the intraclass correlation coefficient (ICC) between predicted and ground truth values.

    Args:
        preds (list or ndarray): List of predicted values.
        GTs (list or ndarray): List of ground truth values.

    Returns:
        float: Intraclass correlation
    """
    assert len(preds) == len(GTs), "Predicted and ground truth values must have the same length."
    n = len(preds)
    mean_pred = np.mean(preds)
    mean_GT = np.mean(GTs)
    SSB = n * (mean_pred - mean_GT)**2
    SSW = np.sum((preds - mean_pred)**2) + np.sum((GTs - mean_GT)**2)
    ICC = (SSB / (SSB + SSW))
    return ICC

def Pearson_correlation_coefficient(preds, GTs):
    """
    Calculate the Pearson correlation coefficient between predicted and ground truth values.

    Args:
        preds (list or ndarray): List of predicted values.
        GTs (list or ndarray): List of ground truth values.

    Returns:
        float: Pearson correlation coefficient
    """
    assert len(preds) == len(GTs), "Predicted and ground truth values must have the same length."
    n = len(preds)
    mean_pred = np.mean(preds)
    mean_GT = np.mean(GTs)
    num = np.sum((preds - mean_pred) * (GTs - mean_GT))
    den = np.sqrt(np.sum((preds - mean_pred)**2) * np.sum((GTs - mean_GT)**2))
    r = num / den
    return r

def coefficient_of_variation(preds):
    """
    Calculate the coefficient of variation of the predicted values.

    Args:
        preds (list or ndarray): List of predicted values.

    Returns:
        float: Coefficient of variation
    """
    mean_pred = np.mean(preds)
    std_pred = np.std(preds)
    CV = std_pred / mean_pred
    return CV

def coefficient_of_variation_copliot(preds):
    """
    Calculate the coefficient of variation of the predicted values.

    Args:
        preds (list or ndarray): List of predicted values.

    Returns:
        float: Coefficient of variation
    """
    mean_pred = np.mean(preds)
    std_pred = np.std(preds)
    CV = std_pred / mean_pred
    return CV

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
def linear_regression(preds, GTs):
    """
    Calculate the linear regression between predicted and ground truth values.

    Args:
        preds (list or ndarray): List of predicted values.
        GTs (list or ndarray): List of ground truth values.

    Returns:
        float: Linear regression
    """
    assert len(preds) == len(GTs), "Predicted and ground truth values must have the same length."
    reg_model = LinearRegression().fit(preds, GTs)
    f_val, p_val = f_regression(preds, GTs.flatten())  # Perform F-test
    
    R2 = reg_model.score(preds, GTs)
    R = np.sqrt(R2)

    return_dict = {
        'reg_model': reg_model,
        'R^2': R2,
        'R': R,
        'coefficients': reg_model.coef_,
        'intercept': reg_model.intercept_,
        'f_val': f_val,
        'p_val': p_val,
    }
    return return_dict

def Bland_Altman_plot(preds, GTs):
    """
    Generate a Bland-Altman plot between predicted and ground truth values.

    Args:
        preds (list or ndarray): List of predicted values.
        GTs (list or ndarray): List of ground truth values.

    Returns:
        dict: Bland-Altman plot data
    """
    assert len(preds) == len(GTs), "Predicted and ground truth values must have the same length."
    mean = np.mean([preds, GTs], axis=0)
    diff = preds - GTs
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff

    return_dict = {
        'mean': mean,
        'diff': diff,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'upper_limit': upper_limit,
        'lower_limit': lower_limit,
    }
    return return_dict

def eval_strains(strain_preds, strain_GTs, eval_config=None):
    """
    Evaluate the strains predicted by a model against the ground truth strains.

    Args:
        strains_pred (list): List of predicted strains.
        strain_GT (list): List of ground truth strains.
        eval_config (dict, optional): Evaluation configuration. Defaults to None.

    Returns:
        list: List of errors between predicted strains and ground truth strains.
    """
    if eval_config is None:
        eval_config = {
            'type': 'absolute',            
        }
    
    if eval_config['type'] == 'absolute':
        errors = abs_error(strain_preds, strain_GTs)
        return_dict = {'errors': errors}
    elif eval_config['type'] in ['ICC', 'intraclass_correlation_coefficient']:
        errors = intraclass_correlation_coefficient(strain_preds, strain_GTs)
        return_dict = {'errors': errors}
    elif eval_config['type'] in ['Pearson', 'Pearson_correlation_coefficient', 'r']:
        errors = Pearson_correlation_coefficient(strain_preds, strain_GTs)
        return_dict = {'errors': errors}
    elif eval_config['type'] in ['CV', 'coefficient_of_variation']:
        errors = coefficient_of_variation(strain_preds)
        return_dict = {'errors': errors}
    elif eval_config['type'] in ['linear_regression']:
        linear_regression_dict = linear_regression(strain_preds, strain_GTs)
        return_dict = linear_regression_dict
    elif eval_config['type'] in ['Bland_Altman_plot']:
        bland_altman_dict = Bland_Altman_plot(strain_preds, strain_GTs)
        return_dict = bland_altman_dict
    else:
        raise ValueError(f"Error type {eval_config['type']} not supported.")

    return return_dict

def eval_strain_matrices(strain_matrix_pred, strain_matrix_GT, ES_frame=None, eval_config=None, n_ignore_frames=5):
    """
    Evaluate the strain matrices predicted by a model against the ground truth strain matrices.

    Args:
        strain_matrix_pred (list): List of predicted strain matrices.
        strain_matrix_GT (list): List of ground truth strain matrices.
        eval_config (dict, optional): Evaluation configuration. Defaults to None.

    Returns:
        list: List of errors between predicted strain matrices and ground truth strain matrices.
    """
    assert strain_matrix_pred.shape == strain_matrix_GT.shape, f"Predicted strain matrix shape {strain_matrix_pred.shape} does not match ground truth strain matrix shape {strain_matrix_GT.shape}."

    n_sectors, n_frames = strain_matrix_pred.shape

    if eval_config is None:
        eval_config = {
            'type':  'absolute',
            'space': 'global', # 'global' or 'segmental'
            'time':  'ES'# 'ES' or 'global'
        }
    if ES_frame is None:
        ES_frame = n_frames // 2

    # initialize preds and GTs
    if n_ignore_frames > 0:
        strain_preds = strain_matrix_pred[:,:-n_ignore_frames]
        strain_GTs = strain_matrix_GT[:,:-n_ignore_frames]
    else:
        strain_preds = strain_matrix_pred
        strain_GTs = strain_matrix_GT
    
    # slicing over the time dimension
    # print(ES_frame, strain_preds.shape, strain_GTs.shape)
    if eval_config['time'] == 'ES':
        strain_preds = strain_preds[:, ES_frame].reshape(-1, 1)
        strain_GTs = strain_GTs[:, ES_frame].reshape(-1, 1)
    elif eval_config['time'] == 'global':
        pass
    else:
        raise ValueError(f"Time dimension evaluation type {eval_config['time']} not supported.")
    
    # slicing over the space dimension
    if eval_config['space'] == 'global':
        strain_preds = np.mean(strain_preds, axis=0)
        strain_GTs = np.mean(strain_GTs, axis=0)
    elif eval_config['space'] == 'segmental':
        pass
    else:
        raise ValueError(f"Space dimension evaluation type {eval_config['space']} not supported.")

    errors = eval_strains(strain_preds, strain_GTs, eval_config)

    return errors