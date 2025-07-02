import numpy as np

class ModelMetrics:
    """
    Всички изчисления използват само NumPy за съвместимост с останалата част от AI модула.
    """
    
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """
        Изчислява матрица на объркванията за бинарна класификация.
        y_true : numpy.ndarray
            Истински бинарни етикети (0 и 1)
        y_pred : numpy.ndarray
            Предсказани бинарни етикети (0 и 1)
            
        Връща:
        -------
        numpy.ndarray
            2x2 матрица на объркванията [[TN, FP], [FN, TP]]

        TN (True Negative) – моделът правилно е предсказал клас 0, когато истинският клас е бил 0

        FP (False Positive) – моделът грешно е предсказал 1, когато истинският клас е бил 0 (т.нар. „false alarm“)

        FN (False Negative) – моделът грешно е предсказал 0, когато истинският клас е бил 1

        TP (True Positive) – моделът правилно е предсказал клас 1, когато истинският клас е бил 1
            
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        tn = np.sum((y_true == 0) & (y_pred == 0))  # Истински негативи
        fp = np.sum((y_true == 0) & (y_pred == 1))  # Фалшиви позитиви
        fn = np.sum((y_true == 1) & (y_pred == 0))  # Фалшиви негативи
        tp = np.sum((y_true == 1) & (y_pred == 1))  # Истински позитиви
        
        cm = np.array([[tn, fp], [fn, tp]])
        return cm
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """
        Изчислява точност (accuracy).
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(y_true, y_pred):
        """
        Изчислява прецизност (precision).
        """
        cm = ModelMetrics.confusion_matrix(y_true, y_pred)
        tp = cm[1, 1]
        fp = cm[0, 1]
        
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)
    
    @staticmethod
    def recall(y_true, y_pred):
        """
        Изчислява чувствителност (recall).
        """
        cm = ModelMetrics.confusion_matrix(y_true, y_pred)
        tp = cm[1, 1]
        fn = cm[1, 0]
        
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)
    
    @staticmethod
    def f1_score(y_true, y_pred):
        """
        Изчислява F1 метриката.
        """
        precision = ModelMetrics.precision(y_true, y_pred)
        recall = ModelMetrics.recall(y_true, y_pred)
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def log_loss(y_true, y_pred_proba):
        """
        Изчислява логаритмична загуба (log loss / крос-ентропийна загуба).
        """
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        
        epsilon = 1e-15
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        
        loss = -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
        return loss
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred, y_pred_proba=None):
        """
        Изчислява всички метрики наведнъж.
        
        Връща речник със стойности за:
        - точност
        - прецизност
        - чувствителност
        - F1
        - матрица на объркванията
        - логаритмична загуба (ако има вероятности)
        """
        metrics = {
            'accuracy': ModelMetrics.accuracy(y_true, y_pred),
            'precision': ModelMetrics.precision(y_true, y_pred),
            'recall': ModelMetrics.recall(y_true, y_pred),
            'f1_score': ModelMetrics.f1_score(y_true, y_pred),
            'confusion_matrix': ModelMetrics.confusion_matrix(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['log_loss'] = ModelMetrics.log_loss(y_true, y_pred_proba)
        
        return metrics
    
    @staticmethod
    def format_confusion_matrix(cm, class_names=['Circle', 'Square']):
        """
        Форматира матрицата на объркванията за визуализация.
        """
        return {
            'matrix': cm.tolist(),
            'class_names': class_names,
            'total': int(np.sum(cm)),
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1])
        }


class InformationGain:
    """
    Изчисляване на информационна печалба за избор на характеристики и анализ.
    """
    
    @staticmethod
    def entropy(y):
        """
        Изчислява ентропия на бинарен вектор.
        """
        if len(y) == 0:
            return 0
        
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-15))
        return entropy
    
    @staticmethod
    def information_gain(X, y, feature_idx, threshold=None):
        """
        Изчислява информационната печалба за дадена характеристика.
        """
        if threshold is None:
            threshold = np.median(X[:, feature_idx])
        
        parent_entropy = InformationGain.entropy(y)
        
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        left_y = y[left_mask]
        right_y = y[right_mask]
        
        n_total = len(y)
        n_left = len(left_y)
        n_right = len(right_y)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        weighted_entropy = (n_left / n_total) * InformationGain.entropy(left_y) + \
                           (n_right / n_total) * InformationGain.entropy(right_y)
        
        information_gain = parent_entropy - weighted_entropy
        return information_gain
    
    @staticmethod
    def analyze_feature_importance(X, y, top_k=20):
        """
        Анализира важността на характеристиките чрез информационна печалба.
        """
        n_features = X.shape[1]
        feature_gains = []
        
        for i in range(n_features):
            gain = InformationGain.information_gain(X, y, i)
            feature_gains.append((i, gain))
        
        feature_gains.sort(key=lambda x: x[1], reverse=True)
        top_features = feature_gains[:top_k]
        
        top_features_formatted = []
        for idx, gain in top_features:
            if n_features == 784:
                row = idx // 28
                col = idx % 28
                pixel_info = f"Пиксел ({row}, {col})"
            else:
                pixel_info = f"Характеристика {idx}"
            
            top_features_formatted.append({
                'feature_idx': idx,
                'feature_name': pixel_info,
                'information_gain': gain
            })
        
        return {
            'top_features': top_features_formatted,
            'all_gains': [gain for _, gain in feature_gains]
        }
