from logging import warning
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prior.data_prior_config import *

DEFAULT_FIXED_HP = {
    # SCMPrior
    "mix_probs": (0.7, 0.3),
    # TreeSCM
    "tree_model": "xgboost",
    "tree_depth_lambda": 0.5,
    "tree_n_estimators_lambda": 0.5,
    # Reg2Cls
    "balanced": False,
    "multiclass_ordered_prob": 0.0,
    "cat_prob": 0.2,
    "max_categories": float("inf"),
    "scale_by_max_features": False,
    "permute_features": True,
    "permute_labels": True,
}

DEFAULT_SAMPLED_HP = {
    # Reg2Cls
    "multiclass_type": {"distribution": "meta_choice", "choice_values": ["value", "rank"]},
    # MLPSCM
    "mlp_activations": {
        "distribution": "meta_choice_mixed",
        "choice_values": get_activations(random=True, scale=True, diverse=True),
    },
    "block_wise_dropout": {"distribution": "meta_choice", "choice_values": [True, False]},
    "mlp_dropout_prob": {"distribution": "meta_beta", "scale": 0.9, "min": 0.1, "max": 5.0},
    # MLPSCM and TreeSCM
    "is_causal": {"distribution": "meta_choice", "choice_values": [True, False]},
    "num_causes": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 12,
        "min_mean": 1,
        "round": True,
        "lower_bound": 1,
    },
    "y_is_effect": {"distribution": "meta_choice", "choice_values": [True, False]},
    "in_clique": {"distribution": "meta_choice", "choice_values": [True, False]},
    "sort_features": {"distribution": "meta_choice", "choice_values": [True, False]},
    "num_layers": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 6,
        "min_mean": 1,
        "round": True,
        "lower_bound": 2,
    },
    "hidden_dim": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 130,
        "min_mean": 5,
        "round": True,
        "lower_bound": 4,
    },
    "init_std": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 10.0,
        "min_mean": 0.01,
        "round": False,
        "lower_bound": 0.0,
    },
    "noise_std": {
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 0.3,
        "min_mean": 0.0001,
        "round": False,
        "lower_bound": 0.0,
    },
    "sampling": {"distribution": "meta_choice", "choice_values": ["normal", "mixed", "uniform"]},
    "pre_sample_cause_stats": {"distribution": "meta_choice", "choice_values": [True, False]},
    "pre_sample_noise_std": {"distribution": "meta_choice", "choice_values": [True, False]},
}


def get_optimizer_config():
    optimizer = {
        "aggregate_k_gradients": 1,
        "learning_rate": 0.00003,
        "epochs": 4000,
        "train_mixed_precision": True,
        'stop_after_epochs': None,
        'reduce_lr_on_spike': False,
        'warmup_steps': None,
        'learning_rate_schedule': 'cosine',
        'first_cycle_steps': 60000,
        'later_cycle_steps': 20000,
        'min_lr': 1e-8,
        'adam_beta1': 0.9,
        'spike_tolerance': 4,
        'weight_decay': 0.0,
        'lr_decay': 0.99,
        'adaptive_batch_size': True,
        'validate_every_n_steps': None,
    }
    return {'optimizer': optimizer}


def get_transformer_config():
    transformer = {
        "emsize": 512,
        "nlayers": 12,
        "dropout": 0.0,
        "nhid_factor": 2,
        'nhead': 512 // 128,
        'init_method': None,
        'recompute_attn': True,
        'pre_norm': False,
        'y_encoder': "one_hot",
        'classification_task': True,
        'efficient_eval_masking': True,
        'input_normalization': False,
        'tabpfn_zero_weights': True,
    #    'model': 'standard_attention',
    }
    return {'transformer': transformer}

def get_prior_config(max_features=100, n_samples=1024+128, mlp_prior_type='default'):
    """"
    Returns the configuration parameters for the tabular multiclass wrapper.
    """

    prior = {
        "num_features": max_features,
        "n_samples": n_samples,
        "eval_positions": [n_samples * 0.95],
        'heterogeneous_batches': False,
        'multiclass_loss_type': 'nono',  # 'compatible'
        'prior_type': 'prior_bag',
        'prior_bag': {'prior_bag_exp_weights_1': {'distribution': 'uniform', 'min': 2.0, 'max': 10.0}},
        'batch_size_per_subgroup': 4}



    if mlp_prior_type == 'default':
        mlp_prior_config = mothernet_mlp_config()
    elif mlp_prior_type == 'tabiclv1' or mlp_prior_type == 'tabiclv2':
        mlp_prior_config = tabiclv1_mlp_config()
        if mlp_prior_type == 'tabiclv2':
            warning("Using tabiclv2 prior, but will run from another branch")
    else:
        raise ValueError(f"Unknown MLP prior type: {mlp_prior_type}")

    prior['mlp'] = mlp_prior_config

    gp_prior_config = {
        'outputscale': {'distribution': 'log_uniform', 'min': 1e-5, 'max': 8},
        'lengthscale': {'distribution': 'log_uniform', 'min': 1e-5, 'max': 8},
        'noise': {'distribution': 'meta_choice', 'choice_values': [0.00001, 0.0001, 0.01]},
        "sampling": 'normal',  # hp.choice('sampling', ['mixed', 'normal']), # uniform
    }

    prior['gp'] = gp_prior_config

    tree_prior_config = {
        # TreeSCM internally hardcodes: is_causal=False, num_layers=[1,2], hidden_dim=[3,9]
        "tree_model": "xgboost",
        "tree_depth_lambda": 0.5,
        "tree_n_estimators_lambda": 0.5,
        "sampling": {'distribution': 'meta_choice', 'choice_values': ['normal', 'mixed', 'uniform']},
        "pre_sample_cause_stats": {"distribution": "meta_choice", "choice_values": [True, False]},
        "pre_sample_noise_std": {"distribution": "meta_choice", "choice_values": [True, False]},
    }

    prior['tree'] = tree_prior_config

    prior['step_function'] = {
        'max_steps': 1,
        'sampling': 'uniform',
    }

    max_num_classes = 10
    classsification_prior = {
        "max_num_classes": max_num_classes,
        "num_classes": {'distribution': 'uniform_int', 'min': 2, 'max': max_num_classes},
        # "noise_type": "Gaussian",  # NN unused?!
        "balanced": False,
        'output_multiclass_ordered_p': 0.,
        'multiclass_max_steps': 10,
        "multiclass_type": 'rank',
        'categorical_feature_p': .2,  # diff: .0
        'nan_prob_no_reason': 0.0,
        'nan_prob_a_reason': 0.0,
        'set_value_to_nan': .9,
        'num_features_sampler': 'uniform',
        'pad_zeros': True,
        'feature_curriculum': False,
    }
    prior['classification'] = classsification_prior

    dataloader = {
        "batch_size": 8,
        "total_steps": None,
        'min_eval_pos': 2,
        'random_n_samples': 0,
        'n_test_samples': 0,
    }
    
    openmlloader = {
        'valid_data': 'old',
        'max_samples': float('inf'),
        'pca': False,
        'val_csv_path': os.path.normpath(os.path.join(os.path.dirname(__file__), 'val_dataset', 'openml_old_datasets_val.csv')),
        'val_pkl_path': os.path.normpath(os.path.join(os.path.dirname(__file__), 'val_dataset', 'openml_old_datasets_val.pkl')),
    }

    prior['boolean'] = {
        'max_fraction_uninformative': 0.5,
        'p_uninformative': 0.5
    }

    return {'prior': prior, 'dataloader': dataloader, 'openmlloader': openmlloader}


def get_mothernet_config():
    return {'mothernet': {
        'weight_embedding_rank': 32,
        'low_rank_weights': True,
        'predicted_hidden_layer_size': 512,
        'predicted_activation': 'relu',
        'decoder_type': "class_average",
        'decoder_embed_dim': 1024,
        'predicted_hidden_layers': 2,
        'decoder_hidden_layers': 1,
        'decoder_hidden_size': 2048,
        'decoder_activation': 'gelu',
        'predictor_type': 'MLP',}
    }


def get_shared_defaults(encoder_type = 'transformer', **kwargs):
    config = get_prior_config(**kwargs)
    config.update(get_optimizer_config())
    if encoder_type == 'transformer':
        config.update(get_transformer_config())
    else:
        raise ValueError(f"Unknown encoder type {encoder_type}")
    return config


def get_mothernet_default_config(**kwargs):
    config = get_shared_defaults(**kwargs)
    config.update(get_mothernet_config())
    return config


def get_model_default_config(model_type, model = None, **kwargs):
    if model_type == 'mothernet':
        config = get_mothernet_default_config(**kwargs)
    else:
        raise ValueError(f"Unknown model type {model_type}")
    config['model_type'] = model_type
    return config
