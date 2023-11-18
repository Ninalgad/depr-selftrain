import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def evaluate(model, val_loader, device, debug=False):
    model.eval()

    prediction, labels = np.array([], 'float32'), np.array([], 'float32')
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch.pop('labels').detach().cpu().numpy()

        logits = model(**batch).logits

        p = logits.detach().cpu().numpy()

        p = np.argmax(p, axis=-1)
        y = np.argmax(y, axis=-1)

        prediction = np.append(prediction, p)
        labels = np.append(labels, y)

        if debug:
            break

    results = {
        'macro f1-score': f1_score(labels, prediction, average='macro'),
        'micro f1-score': f1_score(labels, prediction, average='micro'),
        'weighted f1-score': f1_score(labels, prediction, average='weighted'),
        'accuracy': accuracy_score(labels, prediction),
        'precision': precision_score(labels, prediction, average='macro'),
        'recall': recall_score(labels, prediction, average='macro')
    }
    return results
