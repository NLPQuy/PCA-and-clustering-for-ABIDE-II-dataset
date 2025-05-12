import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

def calculate_cluster_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for clustering results against true labels.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted cluster labels
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Convert labels to binary 0/1 for Cancer/Normal problems
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    
    # For clustering where labels might be arbitrary:
    # We need to find the best mapping from cluster labels to true labels
    
    # Calculate confusion matrix - rows are true labels, columns are predicted labels
    cm = confusion_matrix(y_true, y_pred)
    
    # If we have exactly two clusters and two classes:
    if len(unique_true) == 2 and len(unique_pred) == 2:
        # Check which mapping gives higher accuracy:
        # Option 1: 0->0, 1->1 (diagonal)
        acc1 = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
        
        # Option 2: 0->1, 1->0 (anti-diagonal)
        acc2 = (cm[0, 1] + cm[1, 0]) / np.sum(cm)
        
        # Choose the better mapping
        if acc1 >= acc2:
            # Keep original labels
            y_mapped = y_pred
            accuracy = acc1
        else:
            # Flip the labels
            y_mapped = 1 - y_pred
            accuracy = acc2
        
        # Calculate precision, recall, F1 with zero_division=0
        precision = precision_score(y_true, y_mapped, zero_division=0)
        recall = recall_score(y_true, y_mapped, zero_division=0)
        f1 = f1_score(y_true, y_mapped, zero_division=0)
    else:
        # For more complex cases with more than 2 clusters or classes,
        # use a more general approach
        
        # Create all possible mappings from cluster labels to true labels
        from itertools import permutations
        all_true_labels = list(range(len(unique_true)))
        
        best_acc = -1
        best_mapped = None
        
        if len(unique_pred) <= len(unique_true) and len(unique_pred) <= 8:  # Limit to avoid too many permutations
            # If there are fewer clusters than true classes, 
            # we can try all mappings
            for perm in permutations(all_true_labels, len(unique_pred)):
                # Create mapping
                mapping = dict(zip(range(len(unique_pred)), perm))
                
                # Map predicted labels
                y_mapped = np.zeros_like(y_pred)
                for i, cluster in enumerate(unique_pred):
                    y_mapped[y_pred == cluster] = mapping[i]
                
                # Calculate accuracy
                acc = accuracy_score(y_true, y_mapped)
                
                if acc > best_acc:
                    best_acc = acc
                    best_mapped = y_mapped
        else:
            # Too many permutations, use a greedy approach
            # Assign each cluster to the class that is most common in that cluster
            y_mapped = np.zeros_like(y_pred)
            for cluster in unique_pred:
                mask = (y_pred == cluster)
                if np.sum(mask) > 0:
                    counts = np.bincount(y_true[mask].astype(int), minlength=len(unique_true))
                    most_common = np.argmax(counts)
                    y_mapped[mask] = most_common
            
            best_mapped = y_mapped
            best_acc = accuracy_score(y_true, best_mapped)
        
        # Calculate metrics with the best mapping
        accuracy = best_acc
        precision = precision_score(y_true, best_mapped, average='weighted', zero_division=0)
        recall = recall_score(y_true, best_mapped, average='weighted', zero_division=0)
        f1 = f1_score(y_true, best_mapped, average='weighted', zero_division=0)
    
    # Calculate additional clustering metrics
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ari': ari,
        'nmi': nmi
    }

def print_metrics(metrics):
    """
    Print metrics in a formatted way.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    if 'ari' in metrics:
        print(f"Adjusted Rand Index: {metrics['ari']:.4f}")
    if 'nmi' in metrics:
        print(f"Normalized Mutual Information: {metrics['nmi']:.4f}") 