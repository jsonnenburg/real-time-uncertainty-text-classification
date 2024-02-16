import numpy as np
import tensorflow as tf


def get_student_predictive_distribution_info(model, eval_data, n=50, num_samples=500) -> dict:
    eval_data = eval_data.unbatch().take(num_samples).batch(32)

    total_logits = []
    total_mean_logits = []  # mean prediction (logit space)
    total_mean_variances = []  # aleatoric uncertainty
    total_variances = []  # epistemic uncertainty (?) - variance of predictive distribution
    total_labels = []
    total_uncertainties = []

    for batch in eval_data:
        features, labels = batch
        outputs = model.mc_dropout_predict(features, n=n)
        logits = outputs['logits']
        mean_predictions = outputs['mean_predictions']
        mean_variances = outputs['mean_variances']
        var_predictions = outputs['var_predictions']
        total_uncertainty = outputs['total_uncertainty']
        total_logits.append(logits.numpy())
        total_mean_logits.extend(mean_predictions.numpy())
        total_mean_variances.extend(mean_variances.numpy())
        total_variances.extend(var_predictions.numpy())
        total_uncertainties.extend(total_uncertainty.numpy())
        total_labels.extend(labels.numpy())

    total_logits = np.concatenate(total_logits, axis=1)
    transposed_logits = list(zip(*total_logits))
    transposed_logits = [item for item in transposed_logits]
    # Convert each tuple of arrays into a single list of raw values
    transposed_logits_raw = [[value for array in tup for value in array.tolist()] for tup in transposed_logits]

    if total_mean_logits and total_labels:
        all_labels = np.array(total_labels)
        mean_prob_predictions_np = tf.nn.sigmoid(total_mean_logits).numpy().reshape(all_labels.shape)
        mean_class_predictions_np = mean_prob_predictions_np.round(0).astype(int)
        mean_variances_np = np.array(total_mean_variances).reshape(all_labels.shape)
        total_uncertainties_np = np.array(total_uncertainties).reshape(all_labels.shape)
        labels_np = all_labels

        return {
            "logits": transposed_logits_raw,
            "y_true": labels_np.astype(int).tolist(),
            "y_pred": mean_class_predictions_np.tolist(),
            "y_prob": mean_prob_predictions_np.tolist(),
            "predictive_variance": mean_variances_np.tolist(),
            "epistemic_uncertainty": (total_uncertainties_np - mean_variances_np).tolist(),
            "total_uncertainty": total_uncertainties_np.tolist()
        }


# for some test sequences, save mc dropout predictive samples (all samples, as well as mean and variance), aleatoric and epistemic uncertainty (epistemic as total uncertainty - aleatoric)
def get_teacher_predictive_distribution_info(model, eval_data, n=50, num_samples=500) -> dict:
    eval_data = eval_data.unbatch().take(num_samples).batch(32)

    total_logits = []
    total_mean_logits = []  # mean prediction (logit space)
    total_mean_variances = []  # aleatoric uncertainty
    total_variances = []  # epistemic uncertainty (?) - variance of predictive distribution
    total_labels = []
    total_uncertainties = []

    for batch in eval_data:
        features, labels = batch
        logits, mean_variances, mean_predictions, var_predictions, total_uncertainty = mc_dropout_predict(model, features, n=n)
        total_logits.append(logits.numpy())
        total_mean_variances.extend(mean_variances.numpy())
        total_mean_logits.extend(mean_predictions.numpy())
        total_variances.extend(var_predictions.numpy())
        total_uncertainties.extend(total_uncertainty.numpy())
        total_labels.extend(labels.numpy())

    total_logits = np.concatenate(total_logits, axis=1)
    transposed_logits = list(zip(*total_logits))
    transposed_logits = [item for item in transposed_logits]
    # Convert each tuple of arrays into a single list of raw values
    transposed_logits_raw = [[value for array in tup for value in array.tolist()] for tup in transposed_logits]

    if total_mean_logits and total_labels:
        all_labels = np.array(total_labels)
        mean_prob_predictions_np = tf.nn.sigmoid(total_mean_logits).numpy().reshape(all_labels.shape)
        mean_class_predictions_np = mean_prob_predictions_np.round(0).astype(int)
        mean_variances_np = np.array(total_mean_variances).reshape(all_labels.shape)
        total_uncertainties_np = np.array(total_uncertainties).reshape(all_labels.shape)
        labels_np = all_labels

        return {
            "logits": transposed_logits_raw,
            "y_true": labels_np.astype(int).tolist(),
            "y_pred": mean_class_predictions_np.tolist(),
            "y_prob": mean_prob_predictions_np.tolist(),
            "predictive_variance": mean_variances_np.tolist(),
            "epistemic_uncertainty": (total_uncertainties_np - mean_variances_np).tolist(),
            "total_uncertainty": total_uncertainties_np.tolist()
        }
