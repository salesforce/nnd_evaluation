import os, json, utils_misc, requests
from datasets import load_dataset
from collections import Counter

# Based on this paper: https://arxiv.org/abs/2111.09525
# Codebase is: https://github.com/tingofurro/summac

CNNDM, cnndm_id2article = None, {}

def get_cnndm_document(aid):
    global CNNDM, cnndm_id2article
    if CNNDM is None:
        CNNDM = load_dataset("cnn_dailymail", "3.0.0")
        cnndm_id2article = {}
        for cut in ["test", "validation"]:
            cnndm_id2article.update({d["id"]: d["article"] for d in CNNDM[cut]})
    return cnndm_id2article[aid]

def load_summeval(dataset_folder, cut="val", key_focus="consistency"):
    assert key_focus in ["consistency", "coherence", "fluency", "relevance"]
    # SummEval: Re-evaluating Summarization Evaluation [https://arxiv.org/abs/2007.12626]
    # Data files downloaded from the Github repository: https://github.com/Yale-LILY/SummEval
    raw_dataset = []

    fn = os.path.join(dataset_folder, "model_annotations.aligned.scored.jsonl")
    if not os.path.exists(dataset_folder):
        print("==== SummEval dataset not found, downloading from scratch")
        os.makedirs(dataset_folder)

        # From the 4/19/2020 update on the README: https://github.com/Yale-LILY/SummEval
        utils_misc.download_file_from_google_drive("1d2Iaz3jNraURP1i7CfTqPIj8REZMJ3tS", fn)

    with open(fn, "r") as f:
        for line in f:
            raw_dataset.append(json.loads(line))

    clean_dataset = []
    for i, d in enumerate(raw_dataset):
        c = "val" if i % 2 == 0 else "test"
        _, _, article_id = d["id"].split("-")
        document = get_cnndm_document(article_id)
        annotations = d["expert_annotations"]

        consistencies = [a[key_focus] for a in annotations]
        final_label = 1 if len([cons for cons in consistencies if cons==5]) > len(annotations)/2 else 0

        # annotations = [1 if cons == 5 else 0 for cons in consistencies]
        annotations = consistencies
        error_type = "no error" if final_label == 1 else "error"

        clean_dataset.append({"document": document, "claim": d["decoded"], "label": final_label, "model_name": d["model_id"], "cnndm_id": d["id"], "cut": c, "annotations": annotations, "dataset": "summeval", "origin": "cnndm", "error_type": error_type})
    final_dataset = [d for d in clean_dataset if d["cut"] == cut]
    return final_dataset

def load_frank(dataset_folder, cut="val"):
    # FRANK: Factuality Evaluation Benchmark [https://aclanthology.org/2021.naacl-main.383.pdf]
    # Files downloaded from the Github repository: https://github.com/artidoro/frank

    if not os.path.exists(dataset_folder):
        print("==== Frank dataset not found, downloading from scratch")
        os.makedirs(dataset_folder)

        fns = ["human_annotations_sentence.json", "validation_split.txt", "test_split.txt"]
        for fn in fns:
            data = requests.get("https://raw.githubusercontent.com/artidoro/frank/main/data/%s" % fn)
            with open(os.path.join(dataset_folder, fn), "w") as f:
                f.write(data.text)

    raw_file = os.path.join(dataset_folder, "human_annotations_sentence.json")
    val_hash_file = os.path.join(dataset_folder, "validation_split.txt")
    test_hash_file = os.path.join(dataset_folder, "test_split.txt")
    with open(val_hash_file if cut=="val" else test_hash_file, "r") as f:
        valid_hashes = set([line.strip() for line in f])

    with open(raw_file, "r") as f:
        raw_dataset = json.load(f)
    dataset = []
    for d in raw_dataset:
        article = d["article"]
        origin = "cnndm" if len(d["hash"]) >= 40 else "xsum"

        if d["hash"] not in valid_hashes:
            continue

        summ_labels = []
        annotator_labels = {}
        for annot in d["summary_sentences_annotations"]:
            annot_vals = [an for ans in annot.values() for an in ans]
            noerror_count = len([an for an in annot_vals if an=="NoE"])
            label = 1 if noerror_count >= 2 else 0
            summ_labels.append(label)
            for anno_name, anno in annot.items():
                if anno_name not in annotator_labels:
                    annotator_labels[anno_name] = []
                annotator_labels[anno_name] += anno

        annotations = [1 if all(a=="NoE" for a in annos) else 0 for annos in annotator_labels.values()]
        label = 0 if any(sl==0 for sl in summ_labels) else 1

        error_type = "NoE"
        if label == 0:
            errors = [anno for annos in annotator_labels.values() for anno in annos if anno != "NoE"]
            error_type = Counter(errors).most_common(1)[0][0]

        summary = d["summary"]
        dataset.append({"document": article, "claim": summary, "label": label, "cut": cut, "hash": d["hash"], "model_name": d["model_name"], "annotations": annotations, "dataset": "frank", "origin": origin, "error_type": error_type})
    return dataset
