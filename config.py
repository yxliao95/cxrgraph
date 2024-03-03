# The config for the relation model (pipe2_re_tokaux_sent.py)
class pipe2_re_tokaux_sent:
    # Best result achieved at seed=33 for ace05, 32 for scierc, 35 for radgraph, 23 for cxrgraph
    seed = 23

    do_train = True
    do_eval = True

    task = "cxrgraph"

    # The predicted data from the entity model. By default, it is in `.../[output_dir]/[output_name]/pred_ner/` (as you set for pipe1_xxx)
    data_dir = {
        "scierc": "/root/autodl-tmp/cxr_structured_report/outputs/scierc_pipe1_ner_tokaux_sent_seed_35/pred_ner",
        "ace05": "/root/autodl-tmp/data/ace05/pred_ner24_sent_90299",
        "radgraph": "/root/autodl-tmp/data/radgraph/pred_ner34_sent_943605",
        "cxrgraph": "/root/autodl-tmp/cxr_structured_report/outputs/cxrgraph_pipe1_ner_tokaux_attrcls_sent_seed_35/pred_ner",
    }

    # The fine-tuned encoder from the entity model. By default, it is in `.../[output_model_dir]/[output_name]/` (as you set for pipe1_xxx)
    model_name_or_path = {
        "scierc": "/root/autodl-tmp/cxr_structured_report/models/scierc_pipe1_ner_tokaux_sent_seed_35",
        "ace05": "/root/autodl-tmp/offline_models/ace05_pipe1_ner_tokaux_doc_seed_24",
        "radgraph": "/root/autodl-tmp/offline_models/radgraph_pipe1_ner_tokaux_sent_seed_34",
        "cxrgraph": "/root/autodl-tmp/cxr_structured_report/models/cxrgraph_pipe1_ner_tokaux_attrcls_sent_seed_35",
    }

    output_name = "test_re"
    output_dir = "/root/autodl-tmp/cxr_structured_report/outputs"
    output_model_dir = "/root/autodl-tmp/cxr_structured_report/models"

    fine_tuned_model_path = ""  # If fine_tuned_model_path is provided, we used this model for eval and pred, otherwise, we use the model from output_model_dir/output_name

    # Dataloader
    max_seq_length = 256
    num_extra_sent_for_candi_ent = 0  # 0: number of extra sentences to get candidate entities

    # Model
    dropout_prob = 0.1

    # train()
    clip_grad_norm = 1.0  # ignored when set to 0
    learning_rate = 2e-5
    weight_decay = 0.0
    warmup_proportion = 0.1

    num_epoch = 20
    grad_accum_steps = 1
    train_batch_size = 8

    print_loss_per_n_step = 150
    eval_per_steps = 2500
    eval_per_epoch = 1

    # evaluate()
    eval_batch_size = 16
    ner_tag_for_eval = "pred_ner"  # pred_ner: inference result / ner: gold label


class pipe1_ner_tokaux_attrcls_doc:
    # Best result achieved at 35 for cxrgraph
    seed = 35

    do_train = True
    do_eval = True
    do_pred = True

    task = "cxrgraph"  # scierc

    data_dir = {
        "cxrgraph": "/root/autodl-tmp/data/cxrgraph/json",
    }

    model_name_or_path = {
        "cxrgraph": "/root/autodl-tmp/offline_models/biomedbert-base-uncased",
    }

    output_name = "test_ner"
    output_dir = "/root/autodl-tmp/cxr_structured_report/outputs"
    output_model_dir = "/root/autodl-tmp/cxr_structured_report/models"

    fine_tuned_model_path = ""  # If fine_tuned_model_path is provided, we used this model for eval and pred, otherwise, we use the model from output_model_dir/output_name

    # Dataloader
    max_seq_length = 512
    valid_seq_length = 384  # 当doc subtoks数量大于max_seq_length时，会按照valid_seq_length拆分doc，并填充至max_seq_length
    max_span_length = 8

    # Model
    width_embedding_dim = 150
    dropout_prob = 0.1

    # train()
    encoder_learning_rate = 1e-5
    ner_learning_rate = 5e-4
    weight_decay = 0.01
    warmup_proportion = 0.1
    clip_grad_norm = 0  # ignored when set to 0

    num_epoch = 100
    grad_accum_steps = 1
    train_batch_size = 1

    print_loss_per_n_step = 25
    eval_per_steps = 2000
    eval_per_epoch = 1

    # evaluate()
    eval_batch_size = 1


class pipe1_ner_tokaux_attrcls_sent:
    # Best result achieved at seed=35 for cxrgraph
    seed = 35

    do_train = True
    do_eval = True
    do_pred = True

    task = "cxrgraph"  # scierc

    data_dir = {
        "cxrgraph": "/root/autodl-tmp/data/cxrgraph/json",
    }

    model_name_or_path = {
        "cxrgraph": "/root/autodl-tmp/offline_models/biomedbert-base-uncased",
    }

    output_name = "test_ner"
    output_dir = "/root/autodl-tmp/cxr_structured_report/outputs"
    output_model_dir = "/root/autodl-tmp/cxr_structured_report/models"

    fine_tuned_model_path = ""  # If fine_tuned_model_path is provided, we used this model for eval and pred, otherwise, we use the model from output_model_dir/output_name

    # Dataloader
    max_seq_length = 512
    max_span_length = 8

    # Model
    width_embedding_dim = 150
    dropout_prob = 0.1

    # train()
    encoder_learning_rate = 1e-5
    ner_learning_rate = 5e-4
    weight_decay = 0.01
    warmup_proportion = 0.1
    clip_grad_norm = 0  # ignored when set to 0

    num_epoch = 100
    grad_accum_steps = 1
    train_batch_size = 16

    print_loss_per_n_step = 25
    eval_per_steps = 2000
    eval_per_epoch = 1

    # evaluate()
    eval_batch_size = 32


class pipe1_ner_tokaux_doc:
    # Best result achieved at seed=24 for ace05, 35 for scierc, 34 for radgraph
    seed = 34

    do_train = True
    do_eval = True
    do_pred = True

    task = "ace05"  # scierc

    data_dir = {
        "scierc": "/root/autodl-tmp/data/scierc_data/processed_data/json",
        "ace05": "/root/autodl-tmp/data/ace05/json",
        "radgraph": "/root/autodl-tmp/data/radgraph/json",
    }

    model_name_or_path = {
        "scierc": "/root/autodl-tmp/offline_models/scibert_scivocab_uncased",
        "ace05": "/root/autodl-tmp/offline_models/bert-base-uncased",
        "radgraph": "/root/autodl-tmp/offline_models/biomedbert-base-uncased",
    }

    output_name = "test_ner"
    output_dir = "/root/autodl-tmp/cxr_structured_report/outputs"
    output_model_dir = "/root/autodl-tmp/cxr_structured_report/models"

    fine_tuned_model_path = ""  # If fine_tuned_model_path is provided, we used this model for eval and pred, otherwise, we use the model from output_model_dir/output_name

    # Dataloader
    max_seq_length = 512
    valid_seq_length = 384  # 当doc subtoks数量大于max_seq_length时，会按照valid_seq_length拆分doc，并填充至max_seq_length
    max_span_length = 8

    # Model
    width_embedding_dim = 150
    dropout_prob = 0.1

    # train()
    encoder_learning_rate = 1e-5
    ner_learning_rate = 5e-4
    weight_decay = 0.01
    warmup_proportion = 0.1
    clip_grad_norm = 0  # ignored when set to 0

    num_epoch = 100
    grad_accum_steps = 1
    train_batch_size = 1

    print_loss_per_n_step = 25
    eval_per_steps = 2000
    eval_per_epoch = 1

    # evaluate()
    eval_batch_size = 1


class pipe1_ner_tokaux_sent:
    # Best result achieved at seed=24 for ace05, 35 for scierc, 34 for radgraph
    seed = 34

    do_train = True
    do_eval = True
    do_pred = True

    task = "radgraph"

    data_dir = {
        "scierc": "/root/autodl-tmp/data/scierc_data/processed_data/json",
        "ace05": "/root/autodl-tmp/data/ace05/json",
        "radgraph": "/root/autodl-tmp/data/radgraph/json",
    }

    model_name_or_path = {
        "scierc": "/root/autodl-tmp/offline_models/scibert_scivocab_uncased",
        "ace05": "/root/autodl-tmp/offline_models/bert-base-uncased",
        "radgraph": "/root/autodl-tmp/offline_models/biomedbert-base-uncased",
    }

    output_name = "test_ner"
    output_dir = "/root/autodl-tmp/cxr_structured_report/outputs"
    output_model_dir = "/root/autodl-tmp/cxr_structured_report/models"

    fine_tuned_model_path = ""  # If fine_tuned_model_path is provided, we used this model for eval and pred, otherwise, we use the model from output_model_dir/output_name

    # Dataloader
    max_seq_length = 512
    max_span_length = 8

    # Model
    width_embedding_dim = 150
    dropout_prob = 0.1

    # train()
    encoder_learning_rate = 1e-5
    ner_learning_rate = 5e-4
    weight_decay = 0.01
    warmup_proportion = 0.1
    clip_grad_norm = 0  # ignored when set to 0

    num_epoch = 100
    grad_accum_steps = 1
    train_batch_size = 16

    print_loss_per_n_step = 25
    eval_per_steps = 2000
    eval_per_epoch = 1

    # evaluate()
    eval_batch_size = 32


TASK_NER_LABEL_LIST = {
    "scierc": ["X", "Method", "OtherScientificTerm", "Task", "Generic", "Material", "Metric"],
    "ace05": ["X", "FAC", "WEA", "LOC", "VEH", "GPE", "ORG", "PER"],
    "radgraph": ["X", "ANAT-DP", "OBS-DP", "OBS-U", "OBS-DA"],
    "cxrgraph": ["X", "Anatomy", "Observation-Present", "Observation-Uncertain", "Observation-Absent", "Location-Attribute"],
}

TASK_ATTR_LABEL_LIST = {
    "cxrgraph": ["X", "Normal", "Abnormal", "Removable", "Essential", "Positive", "Negative", "Unchanged"],
}


TASK_REL_LABEL_INFO_DICT = {
    "scierc": {
        "X": {"label_id": 0, "label_str": "X", "type": "symmetric", "inverse_label_id": 0, "inverse_label_str": "X"},
        "CONJUNCTION": {"label_id": 1, "label_str": "CONJUNCTION", "type": "symmetric", "inverse_label_id": 1, "inverse_label_str": "CONJUNCTION"},
        "COMPARE": {"label_id": 2, "label_str": "COMPARE", "type": "symmetric", "inverse_label_id": 2, "inverse_label_str": "COMPARE"},
        "PART-OF": {"label_id": 3, "label_str": "PART-OF", "type": "asymmetric", "inverse_label_id": 8, "inverse_label_str": "OF-PART"},
        "USED-FOR": {"label_id": 4, "label_str": "USED-FOR", "type": "asymmetric", "inverse_label_id": 9, "inverse_label_str": "FOR-USED"},
        "FEATURE-OF": {"label_id": 5, "label_str": "FEATURE-OF", "type": "asymmetric", "inverse_label_id": 10, "inverse_label_str": "OF-FEATURE"},
        "EVALUATE-FOR": {"label_id": 6, "label_str": "EVALUATE-FOR", "type": "asymmetric", "inverse_label_id": 11, "inverse_label_str": "FOR-EVALUATE"},
        "HYPONYM-OF": {"label_id": 7, "label_str": "HYPONYM-OF", "type": "asymmetric", "inverse_label_id": 12, "inverse_label_str": "OF-HYPONYM"},
        "OF-PART": {"label_id": 8, "label_str": "OF-PART", "type": "inverse", "inverse_label_id": 3, "inverse_label_str": "PART-OF"},
        "FOR-USED": {"label_id": 9, "label_str": "FOR-USED", "type": "inverse", "inverse_label_id": 4, "inverse_label_str": "USED-FOR"},
        "OF-FEATURE": {"label_id": 10, "label_str": "OF-FEATURE", "type": "inverse", "inverse_label_id": 5, "inverse_label_str": "FEATURE-OF"},
        "FOR-EVALUATE": {"label_id": 11, "label_str": "FOR-EVALUATE", "type": "inverse", "inverse_label_id": 6, "inverse_label_str": "EVALUATE-FOR"},
        "OF-HYPONYM": {"label_id": 12, "label_str": "OF-HYPONYM", "type": "inverse", "inverse_label_id": 7, "inverse_label_str": "HYPONYM-OF"},
    },
    "ace05": {
        "X": {"label_id": 0, "label_str": "X", "type": "symmetric", "inverse_label_id": 0, "inverse_label_str": "X"},
        "PER-SOC": {"label_id": 1, "label_str": "PER-SOC", "type": "symmetric", "inverse_label_id": 1, "inverse_label_str": "CONJUNCTION"},
        "ART": {"label_id": 2, "label_str": "ART", "type": "asymmetric", "inverse_label_id": 7, "inverse_label_str": "inverse_ART"},
        "ORG-AFF": {"label_id": 3, "label_str": "ORG-AFF", "type": "asymmetric", "inverse_label_id": 8, "inverse_label_str": "AFF-ORG"},
        "GEN-AFF": {"label_id": 4, "label_str": "GEN-AFF", "type": "asymmetric", "inverse_label_id": 9, "inverse_label_str": "AFF-GEN"},
        "PHYS": {"label_id": 5, "label_str": "PHYS", "type": "asymmetric", "inverse_label_id": 10, "inverse_label_str": "inverse_PHYS"},
        "PART-WHOLE": {"label_id": 6, "label_str": "PART-WHOLE", "type": "asymmetric", "inverse_label_id": 11, "inverse_label_str": "WHOLE-PART"},
        "inverse_ART": {"label_id": 7, "label_str": "inverse_ART", "type": "inverse", "inverse_label_id": 2, "inverse_label_str": "ART"},
        "AFF-ORG": {"label_id": 8, "label_str": "AFF-ORG", "type": "inverse", "inverse_label_id": 3, "inverse_label_str": "ORG-AFF"},
        "AFF-GEN": {"label_id": 9, "label_str": "AFF-GEN", "type": "inverse", "inverse_label_id": 4, "inverse_label_str": "GEN-AFF"},
        "inverse_PHYS": {"label_id": 10, "label_str": "inverse_PHYS", "type": "inverse", "inverse_label_id": 5, "inverse_label_str": "PHYS"},
        "WHOLE-PART": {"label_id": 11, "label_str": "WHOLE-PART", "type": "inverse", "inverse_label_id": 6, "inverse_label_str": "PART-WHOLE"},
    },
    "radgraph": {
        "X": {"label_id": 0, "label_str": "X", "type": "symmetric", "inverse_label_id": 0, "inverse_label_str": "X"},
        "suggestive_of": {"label_id": 1, "label_str": "suggestive_of", "type": "asymmetric", "inverse_label_id": 4, "inverse_label_str": "of_suggestive"},
        "located_at": {"label_id": 2, "label_str": "located_at", "type": "asymmetric", "inverse_label_id": 5, "inverse_label_str": "at_located"},
        "modify": {"label_id": 3, "label_str": "modify", "type": "asymmetric", "inverse_label_id": 6, "inverse_label_str": "inverse_modify"},
        "of_suggestive": {"label_id": 4, "label_str": "of_suggestive", "type": "inverse", "inverse_label_id": 1, "inverse_label_str": "suggestive_of"},
        "at_located": {"label_id": 5, "label_str": "at_located", "type": "inverse", "inverse_label_id": 2, "inverse_label_str": "located_at"},
        "inverse_modify": {"label_id": 6, "label_str": "inverse_modify", "type": "inverse", "inverse_label_id": 3, "inverse_label_str": "modify"},
    },
    "cxrgraph": {
        "X": {"label_id": 0, "label_str": "X", "type": "symmetric", "inverse_label_id": 0, "inverse_label_str": "X"},
        "suggestive_of": {"label_id": 1, "label_str": "suggestive_of", "type": "asymmetric", "inverse_label_id": 5, "inverse_label_str": "of_suggestive"},
        "located_at": {"label_id": 2, "label_str": "located_at", "type": "asymmetric", "inverse_label_id": 6, "inverse_label_str": "at_located"},
        "modify": {"label_id": 3, "label_str": "modify", "type": "asymmetric", "inverse_label_id": 7, "inverse_label_str": "inverse_modify"},
        "part_of": {"label_id": 4, "label_str": "part_of", "type": "asymmetric", "inverse_label_id": 8, "inverse_label_str": "of_part"},
        "of_suggestive": {"label_id": 5, "label_str": "of_suggestive", "type": "inverse", "inverse_label_id": 1, "inverse_label_str": "suggestive_of"},
        "at_located": {"label_id": 6, "label_str": "at_located", "type": "inverse", "inverse_label_id": 2, "inverse_label_str": "located_at"},
        "inverse_modify": {"label_id": 7, "label_str": "inverse_modify", "type": "inverse", "inverse_label_id": 3, "inverse_label_str": "modify"},
        "of_part": {"label_id": 8, "label_str": "of_part", "type": "inverse", "inverse_label_id": 4, "inverse_label_str": "part_of"},
    },
}
