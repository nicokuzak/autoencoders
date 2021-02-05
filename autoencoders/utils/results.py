from sklearn.metrics import precision_recall_fscore_support

def autoencoder_results(error_df, threshold):
    y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
    res = precision_recall_fscore_support(error_df.true_class, y_pred)
    return res[0], res[1], res[2], res[3]


def result_dicts(error_df):
    recall_by_threshold = {}
    precision_by_threshold = {}
    f1_by_threshold = {}
    for i in range(1000):
        threshold = (i+1)/5
        precision, recall, f1, _ = autoencoder_results(error_df, threshold)
        recall_by_threshold[threshold] = recall[1]
        precision_by_threshold[threshold] = precision[1]
        f1_by_threshold[threshold] = f1[1]
    return precision_by_threshold, recall_by_threshold, f1_by_threshold