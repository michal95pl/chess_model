import torch
import os
from utils.logger import Logger
from tqdm import tqdm
from matplotlib import pyplot as plt
from model.chessModel import ChessModel
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import datetime

class ModelEvaluator(Logger):

    def __init__(self, test_data_folder_path, device: torch.device, net):
        super().__init__()

        if not os.path.exists(test_data_folder_path):
            self.error(f"Test data folder path '{test_data_folder_path}' does not exist.")
            return

        self.test_data_folder_path = test_data_folder_path
        self.device = device
        self.net = net

    def save_losses_plot(self, models_folder_path, skip_factor: int = 0, plot_path: str = "losses_plot"):
        plot_path += ".png"
        learn_losses, test_losses = self.__get_losses(self.net, models_folder_path, skip_factor)
        learn_policy_loss, learn_value_loss = zip(*learn_losses)
        test_policy_loss, test_value_loss = zip(*test_losses)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(learn_policy_loss, label='Learn Policy Loss', color='blue')
        plt.plot(test_policy_loss, label='Test Policy Loss', color='orange')
        plt.xlabel('Model Version')
        plt.ylabel('Loss')
        plt.title('Policy Loss Over Model Versions')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(learn_value_loss, label='Learn Value Loss', color='blue')
        plt.plot(test_value_loss, label='Test Value Loss', color='orange')
        plt.xlabel('Model Version')
        plt.ylabel('Loss')
        plt.title('Value Loss Over Model Versions')
        plt.legend()

        plt.show()
        self.info(f"Saved losses plot to {plot_path}")

    @staticmethod
    def skip_elements(data: list, skip_factor):
        # todo: improve skipping (e.g., linear skip)
        return [data[i] for i in range(0, len(data), skip_factor+1)]

    def __get_losses(self, net, models_folder_path: str, skip_factor: int):

        files = {}
        for file_name in os.listdir(models_folder_path):
            data = torch.load(os.path.join(models_folder_path, file_name), weights_only=False)
            timestamp = data.get('timestamp')
            dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
            files[dt] = file_name

        files = sorted(files.items())
        files = [file_name for _, file_name in files]
        if skip_factor > 0:
            files = self.skip_elements(files, skip_factor)
            self.info(f"Skipped elements with factor {skip_factor}.")

        self.info("Sorted model files by timestamp.")

        learn_losses = []
        test_losses = []
        with tqdm(total=len(files), desc="Evaluating models", colour="blue") as pbar:
            for file in files:
                data = torch.load(os.path.join(models_folder_path, file), weights_only=False)
                net.load_state_dict(data['model'])
                learn_losses.append((data['avg_policy_loss'], data['avg_value_loss']))
                test_losses.append(ChessModel(self.device).get_model_loss(net, self.test_data_folder_path))
                pbar.update(1)
        return learn_losses, test_losses


    def save_confusion_matrix(self, plot_path: str = "confusion_matrix"):
        plot_path += ".png"
        y_pred_value, y_true_value, y_pred_policy, y_true_policy = ChessModel(self.device).get_value_network_confusion_matrix(self.net, self.test_data_folder_path)

        cm_value = confusion_matrix(y_true_value, y_pred_value)
        precision_value = precision_score(y_true_value, y_pred_value, zero_division=0)
        recall_value = recall_score(y_true_value, y_pred_value, zero_division=0)
        f1_value = f1_score(y_true_value, y_pred_value, zero_division=0)
        metrics_text = f'Precision: {precision_value:.2f}\nRecall: {recall_value:.2f}\nF1: {f1_value:.2f}'


        plt.imshow(cm_value, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix for Value Network\n' + metrics_text)
        tick_marks = range(2)
        plt.xticks(tick_marks, ['-1', '1'])
        plt.yticks(tick_marks, ['-1', '1'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        for i in range(cm_value.shape[0]):
            for j in range(cm_value.shape[1]):
                plt.text(j, i, cm_value[i, j], horizontalalignment="center", color="white" if cm_value[i, j] > cm_value.max() / 2 else "black")

        plt.savefig(plot_path)
        self.info(f"Saved confusion matrix to {plot_path}")