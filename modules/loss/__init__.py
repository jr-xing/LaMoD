from modules.loss.loss_calculator import LossCalculator
# from modules.loss.loss_calculator_hardcoded import HardCodedLossCalculator
from modules.loss.registration_losses import RegistrationReconstructionLoss

def get_average_performance_dict(performances: list, prefix_level:int=1, new_prefix:str='average/'):
    """
    Compute the average performance of a list of performance dictionaries.

    Args:
        performances (list): A list of performance dictionaries.
        prefix_level (int, optional): The number of levels to be used as prefix for averaging. Defaults to 1.
        new_prefix (str, optional): The new prefix to be added to the averaged performance dictionary. Defaults to 'average/'.

    Returns:
        dict: A dictionary containing the average performance values.

    Example:
        all_perf = [
            {
                "fold0/train/registration_reconstruction": 363.23606872558594,
                "fold0/train/registration_supervision": 0.05973426252603531,
                "fold0/train/TOS_regression": 3081.8114318847656,
                "fold0/train/total_loss": 404.239013671875,
                "fold0/test/registration_reconstruction": 82.3651008605957,
                "fold0/test/registration_supervision": 0.03921514190733433,
                "fold0/test/TOS_regression": 474.12152099609375,
                "fold0/test/total_loss": 94.86380386352539,
            }
        ]

        average_perf = get_average_performance_dict(all_perf, prefix_level=1, new_prefix='average/')
        # Output: {'average/train/registration_reconstruction': 363.23606872558594, 'average/train/registration_supervision': 0.05973426252603531, ...}
    """
    average_performance = {}
    for perf in performances:
        for key, value in perf.items():
            key_split = key.split('/')
            key_split = key_split[prefix_level:]
            key = '/'.join(key_split)
            if key not in average_performance:
                average_performance[key] = []
            average_performance[key].append(value)
    
    for key, value in average_performance.items():
        if isinstance(value[0], (float, int)):
            average_performance[key] = sum(value) / len(value)
    
    # add new prefix to all keys
    if new_prefix != '':
        new_average_performance = {}
        for key, value in average_performance.items():
            new_average_performance[new_prefix + key] = value
        average_performance = new_average_performance

    return average_performance


"""
# Example of performance dictionary
all_perf = [
    {
        "fold0/train/registration_reconstruction": 363.23606872558594,
        "fold0/train/registration_supervision": 0.05973426252603531,
        "fold0/train/TOS_regression": 3081.8114318847656,
        "fold0/train/total_loss": 404.239013671875,
        "fold0/test/registration_reconstruction": 82.3651008605957,
        "fold0/test/registration_supervision": 0.03921514190733433,
        "fold0/test/TOS_regression": 474.12152099609375,
        "fold0/test/total_loss": 94.86380386352539,
        "fold0/final-val/sector_error": 12.746996217757937,
        "fold0/final-test/sector_error": 10.035304701808608
    },
    {
        "fold1/train/registration_reconstruction": 346.00063705444336,
        "fold1/train/registration_supervision": 0.4962502606213093,
        "fold1/train/TOS_regression": 4273.211273193359,
        "fold1/train/total_loss": 958.1714630126953,
        "fold1/test/registration_reconstruction": 110.35323333740234,
        "fold1/test/registration_supervision": 0.03786645457148552,
        "fold1/test/TOS_regression": 1141.4432983398438,
        "fold1/test/total_loss": 163.04611206054688,
        "fold1/final-val/sector_error": 16.18214781746032,
        "fold1/final-test/sector_error": 11.974927455357143
    },
    {
        "fold2/train/registration_reconstruction": 366.8650436401367,
        "fold2/train/registration_supervision": 0.5009422833099961,
        "fold2/train/TOS_regression": 4452.974807739258,
        "fold2/train/total_loss": 982.9262847900391,
        "fold2/registration_reconstruction": 84.9059829711914,
        "fold2/registration_supervision": 0.030394199304282665,
        "fold2/TOS_regression": 1043.0859375,
        "fold2/total_loss": 143.19339752197266,
        "fold2/final-val/sector_error": 16.18214657738095,
        "fold2/final-test/sector_error": 11.145366089143991
    },
    {
        "fold3/train/registration_reconstruction": 338.3963203430176,
        "fold3/train/registration_supervision": 0.5011782823130488,
        "fold3/train/TOS_regression": 3986.15966796875,
        "fold3/train/total_loss": 933.6338806152344,
        "fold3/test/registration_reconstruction": 124.8035659790039,
        "fold3/test/registration_supervision": 0.030895670875906944,
        "fold3/test/TOS_regression": 1468.3635864257812,
        "fold3/test/total_loss": 190.21238708496094,
        "fold3/final-val/sector_error": 16.182152157738095,
        "fold3/final-test/sector_error": 14.412205933263062
    },
    {
        "fold4/train/registration_reconstruction": 363.81214904785156,
        "fold4/train/registration_supervision": 0.5054872510954738,
        "fold4/train/TOS_regression": 3733.1864013671875,
        "fold4/train/total_loss": 915.1871109008789,
        "fold4/test/registration_reconstruction": 99.38775634765625,
        "fold4/test/registration_supervision": 0.026578107848763466,
        "fold4/test/TOS_regression": 1721.3368530273438,
        "fold4/test/total_loss": 208.65057373046875,
        "fold4/final-val/sector_error": 14.412205933263062,
        "fold4/final-test/sector_error": 16.182152157738095
    }
]
"""