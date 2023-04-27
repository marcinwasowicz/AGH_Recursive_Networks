import json

SUFFIX_PREFIX_MAP = {"_n_ary_": "", "_sum_": "child_sum_", "_single_": "single_gate_"}


if __name__ == "__main__":
    for model in ["lstm", "gru", "mgu"]:
        for suffix in SUFFIX_PREFIX_MAP:
            file_name = model + suffix + "f.json"
            model_name = SUFFIX_PREFIX_MAP[suffix] + model
            config = {
                "model_types": [model_name],
                "embeddings": ["fasttext-crawl-300d-2M-subword"],
            }

            with open(file_name, "w+") as f:
                json.dump(config, f)
