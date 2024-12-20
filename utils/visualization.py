import matplotlib.pyplot as plt
from datetime import datetime

class Visualization:
    def __init__(self, y_gt, y_pred, isTrain=True):
        self.y_gt = y_gt
        self.y_pred = y_pred
        self.isTrain = isTrain

    def plot_results(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.isTrain:
            plt.figure(figsize=(10, 5))
            plt.scatter(self.y_gt, self.y_pred, color='blue', label='Pred vs GT')
            plt.plot([self.y_gt.min(), self.y_gt.max()], [self.y_gt.min(), self.y_gt.max()], 'k--', lw=2, label='Ideal Fit')
            plt.xlabel('Actual Preliminary Decision')
            plt.ylabel('Predicted Preliminary Decision')
            plt.title('Train: Actual vs Predicted Preliminary Decision')
            plt.legend()
            plt.savefig(f'./Vis/results_train_{timestamp}.png')
            plt.show()
        else:
            plt.figure(figsize=(10, 5))
            plt.scatter(self.y_gt, self.y_pred, color='blue', label='Pred vs GT')
            plt.plot([self.y_gt.min(), self.y_gt.max()], [self.y_gt.min(), self.y_gt.max()], 'k--', lw=2, label='Ideal Fit')
            plt.xlabel('Actual Preliminary Decision')
            plt.ylabel('Predicted Preliminary Decision')
            plt.title('Test: Actual vs Predicted Preliminary Decision')
            plt.legend()
            plt.savefig(f'./Vis/results_test_{timestamp}.png')
            plt.show()