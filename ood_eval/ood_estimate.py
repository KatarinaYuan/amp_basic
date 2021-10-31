import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibrationDisplay



def try_calibrate():
    '''
    https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_multiclass.html#sphx-glr-auto-examples-calibration-plot-calibration-multiclass-py

    '''
    pass 


def calibration(probs, labels, n_bins=10):
    '''
        Params:
            probs: dict of np.array
    '''
    print("NOTE: this should be carried on Test data, especially Witten test data")
    clf_names = probs.keys()
    # calibration_curve(y_true, y_pred, n_bins=n_bins)

    fig = plt.figure(figsize=(10, 10)) ##
    gs = GridSpec(4, 2) ##
    colors = plt.cm.get_cmap('Dark2')
    ax_calibration_curve = fig.add_subplot(gs[:2, :2]) ##

    # Calibrarion Curve
    calibration_displays = {}
    for i, name in enumerate(clf_names):
        display = CalibrationDisplay.from_predictions(
            labels[name], probs[name][:, 1], n_bins=n_bins, name=name, ax=ax_calibration_curve,
            color=colors(i)
        )
        calibration_displays[name] = display
    
    ax_calibration_curve.grid()
    ax_calibration_curve.set_title('Calibration plots')
    
    # Density Estimation
    grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)] ##
    for i, name in enumerate(clf_names):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])

        ax.hist(
            calibration_displays[name].y_prob, range=(0, 1), bins=10, label=name,
            color=colors(i)
        )
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")
    
    # Brier score

    print("Brier score losses: (the smaller the better)")
    for name in clf_names:
        clf_score = brier_score_loss(labels[name], probs[name][:, 1])
        print("Brier score - {}: {:1.3f}".format(name, clf_score))

    plt.tight_layout()
    plt.savefig('calibration.jpg')