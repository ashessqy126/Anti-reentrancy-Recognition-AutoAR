from sys_recognize import sys_recognition

down_sampling_ratio = 0.7
pooling_ratio = 0.7

def evaluate_performance(vulnerable_graphs, non_vulnerable_graphs):
    embed_dir1 = vulnerable_graphs + '_embedding4'
    embed_dir2 = non_vulnerable_graphs + '_embedding4'
    fp, total_minus = sys_recognition(vulnerable_graphs, down_sampling_ratio, pooling_ratio, embed_dir1)
    tp, total_positive = sys_recognition(non_vulnerable_graphs, down_sampling_ratio, pooling_ratio, embed_dir2)
    TPR = tp / total_positive
    FPR = fp / total_minus
    fn = total_positive - tp
    precision = tp / (fp + tp)
    FNR = fn / total_positive
    return TPR, precision, FPR, FNR


if __name__ == '__main__':
    TPR, precision, FPR, FNR = evaluate_performance('test_vulnerable_graphs', 'test_non_vulnerable_graphs')
    print(f'True Positive Ratio (Recall): {TPR}, Precision: {precision}, '
          f'False Positive Ratio (FPR): {FPR}, False Negative Ratio (FNR): {FNR}')