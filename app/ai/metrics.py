
import numpy as np

class ModelMetrics:
    """
    Comprehensive evaluation metrics for binary classification models.
    All calculations use NumPy only for consistency with the rest of the AI module.
    """
    
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """
        Compute confusion matrix for binary classification.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True binary labels (0s and 1s)
        y_pred : numpy.ndarray
            Predicted binary labels (0s and 1s)
            
        Returns:
        --------
        numpy.ndarray
            2x2 confusion matrix [[TN, FP], [FN, TP]]
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate confusion matrix components
        tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
        fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
        fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
        tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
        
        cm = np.array([[tn, fp], [fn, tp]])
        return cm
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """
        Calculate accuracy score.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True binary labels
        y_pred : numpy.ndarray
            Predicted binary labels
            
        Returns:
        --------
        float
            Accuracy score between 0 and 1
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(y_true, y_pred):
        """
        Calculate precision score.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True binary labels
        y_pred : numpy.ndarray
            Predicted binary labels
            
        Returns:
        --------
        float
            Precision score between 0 and 1
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
        Calculate recall score.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True binary labels
        y_pred : numpy.ndarray
            Predicted binary labels
            
        Returns:
        --------
        float
            Recall score between 0 and 1
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
        Calculate F1 score.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True binary labels
        y_pred : numpy.ndarray
            Predicted binary labels
            
        Returns:
        --------
        float
            F1 score between 0 and 1
        """
        precision = ModelMetrics.precision(y_true, y_pred)
        recall = ModelMetrics.recall(y_true, y_pred)
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def log_loss(y_true, y_pred_proba):
        """
        Calculate log loss (cross-entropy loss).
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True binary labels
        y_pred_proba : numpy.ndarray
            Predicted probabilities
            
        Returns:
        --------
        float
            Log loss value
        """
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        
        loss = -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
        return loss
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred, y_pred_proba=None):
        """
        Calculate all evaluation metrics at once.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True binary labels
        y_pred : numpy.ndarray
            Predicted binary labels
        y_pred_proba : numpy.ndarray, optional
            Predicted probabilities for positive class
            
        Returns:
        --------
        dict
            Dictionary containing all calculated metrics
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
        Format confusion matrix for display.
        
        Parameters:
        -----------
        cm : numpy.ndarray
            2x2 confusion matrix
        class_names : list
            Names for the classes
            
        Returns:
        --------
        dict
            Formatted confusion matrix data
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
    Calculate information gain for feature selection and analysis.
    """
    
    @staticmethod
    def entropy(y):
        """
        Calculate entropy of a binary label array.
        
        Parameters:
        -----------
        y : numpy.ndarray
            Binary labels (0s and 1s)
            
        Returns:
        --------
        float
            Entropy value
        """
        if len(y) == 0:
            return 0
        
        # Calculate probabilities
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-15))
        return entropy
    
    @staticmethod
    def information_gain(X, y, feature_idx, threshold=None):
        """
        Calculate information gain for a specific feature.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Binary labels
        feature_idx : int
            Index of the feature to analyze
        threshold : float, optional
            Threshold for splitting. If None, uses median.
            
        Returns:
        --------
        float
            Information gain value
        """
        if threshold is None:
            threshold = np.median(X[:, feature_idx])
        
        # Calculate parent entropy
        parent_entropy = InformationGain.entropy(y)
        
        # Split data based on threshold
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        left_y = y[left_mask]
        right_y = y[right_mask]
        
        # Calculate weighted entropy after split
        n_total = len(y)
        n_left = len(left_y)
        n_right = len(right_y)
        
        if n_left == 0 or n_right == 0:
            return 0  # No information gain if no split occurs
        
        weighted_entropy = (n_left / n_total) * InformationGain.entropy(left_y) + \
                          (n_right / n_total) * InformationGain.entropy(right_y)
        
        # Information gain = parent entropy - weighted child entropy
        information_gain = parent_entropy - weighted_entropy
        return information_gain
    
    @staticmethod
    def analyze_feature_importance(X, y, top_k=20):
        """
        Analyze feature importance using information gain.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Binary labels
        top_k : int
            Number of top features to return
            
        Returns:
        --------
        dict
            Dictionary with feature importance analysis
        """
        n_features = X.shape[1]
        feature_gains = []
        
        for i in range(n_features):
            gain = InformationGain.information_gain(X, y, i)
            feature_gains.append((i, gain))
        
        # Sort by information gain (descending)
        feature_gains.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k features
        top_features = feature_gains[:top_k]
        
        # For image data, convert indices to pixel coordinates
        top_features_formatted = []
        for idx, gain in top_features:
            # Assuming 28x28 image flattened to 784 features
            if n_features == 784:
                row = idx // 28
                col = idx % 28
                pixel_info = f"Pixel ({row}, {col})"
            else:
                pixel_info = f"Feature {idx}"
            
            top_features_formatted.append({
                'feature_idx': idx,
                'feature_name': pixel_info,
                'information_gain': gain
            })
        
        return {
            'top_features': top_features_formatted,
            'all_gains': [gain for _, gain in feature_gains]
        }
