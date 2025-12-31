import os
import jiwer
import pandas as pd

OCR_ROOT = 'results/ocr/'  # path to OCR results
GT_DIR = '../dataset/data/dataset_split/test/ocr_texts'  # path to ground truth texts

ocr_folders = [f for f in os.listdir(OCR_ROOT) if f.startswith('ocr_')]

summary_data = []

# process each OCR result folder
for folder in ocr_folders:
    try:
        parts = folder.split('_')
        engine = parts[-1]
        model = "_".join(parts[1:-1])
    except:
        continue

    pred_dir = os.path.join(OCR_ROOT, folder)
    print(f"Analyzing {model} with {engine}...")

    total_s, total_d, total_i = 0, 0, 0
    total_len = 0
    confusion_pairs = []

    gt_files = [f for f in os.listdir(GT_DIR) if f.endswith('.txt')]

    for filename in gt_files:
        with open(os.path.join(GT_DIR, filename), 'r', encoding='utf-8') as f:
            truth = f.read().strip().replace('\n', ' ')

        pred_path = os.path.join(pred_dir, filename)
        if os.path.exists(pred_path):
            with open(pred_path, 'r', encoding='utf-8') as f:
                pred = f.read().strip()
        else:
            pred = ""

        if not truth: continue

        # jiwer character-level analysis
        out = jiwer.process_characters(truth, pred)
        total_s += out.substitutions
        total_d += out.deletions
        total_i += out.insertions
        total_len += len(truth)

        # collect confusion pairs
        for chunk in out.alignments[0]:
            if chunk.type == 'substitute':
                ref = truth[chunk.ref_start_idx:chunk.ref_end_idx]
                hyp = pred[chunk.hyp_start_idx:chunk.hyp_end_idx]
                for r, h in zip(ref, hyp): confusion_pairs.append((r, h))
            elif chunk.type == 'delete':
                for char in truth[chunk.ref_start_idx:chunk.ref_end_idx]:
                    confusion_pairs.append((char, '<DEL>'))
            elif chunk.type == 'insert':
                for char in pred[chunk.hyp_start_idx:chunk.hyp_end_idx]:
                    confusion_pairs.append(('<INS>', char))
            elif chunk.type == 'equal':
                ref = truth[chunk.ref_start_idx:chunk.ref_end_idx]
                hyp = pred[chunk.hyp_start_idx:chunk.hyp_end_idx]
                for r, h in zip(ref, hyp): confusion_pairs.append((r, h))

    # calculating character error rate
    cer = (total_s + total_d + total_i) / total_len if total_len > 0 else 1.0

    # save results
    with open(f"results_{model}_{engine}.txt", 'w') as f:
        f.write(f"Model: {model} | Engine: {engine}\n")
        f.write(f"CER: {cer:.4f}\n")
        f.write(f"S: {total_s}\nD: {total_d}\nI: {total_i}\n")

    # save confusion matrices
    pd.DataFrame(confusion_pairs, columns=['Truth', 'Pred']).to_csv(f"confusion_{model}_{engine}.csv", index=False)

    summary_data.append([model, engine, cer, total_s, total_d, total_i])

# save summary
df = pd.DataFrame(summary_data, columns=['Model', 'Engine', 'CER', 'Sub', 'Del', 'Ins'])
df.to_csv('final_evaluation_summary.csv', index=False)
print("Analysis Complete.")
