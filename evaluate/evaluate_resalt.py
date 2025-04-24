# Данные будут в формате json, а именно:
# {
#     "name_1+raund": [resualt_round_1, ..., resualt_round_2],
#     .
#     .
#     .
#     "name_n+raund": [resualt_round_1, ..., resualt_round_2]
# }

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
class EvaluateResalt:
    def __init__(self, name_file_resault):
        self.name_file_resault = name_file_resault

    def evaluate_accuracy_for_round(self):
        ...
    
    def evaluate_accuracy_for_all_round(self):
        ...