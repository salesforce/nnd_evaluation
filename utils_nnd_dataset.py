from utils_summarization import load_summeval, load_frank
import os, csv, itertools, json

# Generic
def generic_dataset_grouper(dataset, group_key):
    groups = {}
    for d in dataset:
        if d[group_key] not in groups:
            groups[d[group_key]] = []
        groups[d[group_key]].append(d)
    return groups

def generic_nnd_generator(dataset, group_key, candidate_key, quality_key, keep_key_as_is=False, document_key="document", common_copy_keys=[], candidate_copy_keys=[], must_include_label=None, minimum_gap=None):
    assert not (minimum_gap is not None and keep_key_as_is), "Both options cannot be used at the same time"

    groups = generic_dataset_grouper(dataset, group_key)
    common_copy_keys += [group_key] # In case it's not in there
    nnd_dataset = []

    for _, group in groups.items():
        for d1, d2 in itertools.combinations(group, 2):
            if d1[quality_key] == d2[quality_key]:
                continue
            if must_include_label is not None and d1[quality_key] != must_include_label and d2[quality_key] != must_include_label:
                continue
            D = {k: d1[k] for k in common_copy_keys}
            D["document"] = d1[document_key]
            if keep_key_as_is:
                label1, label2 = d1[quality_key], d2[quality_key]
            else:
                if minimum_gap is not None and abs(d1[quality_key] - d2[quality_key]) < minimum_gap:
                    continue # Skip if the gap is smaller than specified

                label1 = "success" if d1[quality_key] > d2[quality_key] else "nopass"
                label2 = "success" if d2[quality_key] > d1[quality_key] else "nopass"

            D.update({"gen1": d1[candidate_key], "error1": label1})
            D.update({"%s1" % (k): d1[k] for k in candidate_copy_keys})
            D.update({"gen2": d2[candidate_key], "error2": label2})
            D.update({"%s2" % (k): d2[k] for k in candidate_copy_keys})

            nnd_dataset.append(D)
    return nnd_dataset

# TASK 1: SUMMARIZATION
def load_frank_nnd(dataset_folder, cut="test"):
    # FRANK: Factuality Evaluation Benchmark [https://aclanthology.org/2021.naacl-main.383.pdf]
    dataset = load_frank(dataset_folder, cut=cut)
    pair_dataset = generic_nnd_generator(dataset, group_key="hash", candidate_key="claim", quality_key="error_type", keep_key_as_is=True, document_key="document", common_copy_keys=["origin"], candidate_copy_keys=["model_name"], must_include_label="NoE")
    pair_dataset = [p for p in pair_dataset if p["origin"] == "cnndm"]
    return pair_dataset

def load_summeval_nnd(dataset_folder, cut="test"):
    # SummEval: Re-evaluating Summarization Evaluation [https://arxiv.org/abs/2007.12626]
    summeval_nnd = []
    for key in ["consistency", "coherence", "fluency", "relevance"]:
        summeval_subset = load_summeval(dataset_folder, cut=cut, key_focus=key)
        summeval_nnd_subset = generic_nnd_generator(summeval_subset, group_key="cnndm_id", candidate_key="claim", quality_key="error_type", keep_key_as_is=True, document_key="document", common_copy_keys=["origin"], candidate_copy_keys=["model_name"])
        for d in summeval_nnd_subset:
            if d["error1"] == "error":
                d["error1"] = "%s" % key
            if d["error2"] == "error":
                d["error2"] = "%s" % key
        summeval_nnd += summeval_nnd_subset
    return summeval_nnd

def load_summ_gpt3_nnd(datafolder=".", dataset_type="cnn", min_score_gap=2):
    # News Summarization and Evaluation in the Era of GPT-3 [https://arxiv.org/abs/2209.12356]
    # Data downloaded: https://tagoyal.github.io/zeroshot-news-annotations.html#human

    assert dataset_type in ["cnn", "bbc", "keyword"]
    filename = "human_annotations.zip"
    full_path = os.path.join(datafolder, filename)
    if not os.path.exists(full_path):
        # Download from Github into datafolder
        print("Downloading %s from Github" % (filename))
        os.system("wget https://tagoyal.github.io/zero-shot-explorer/human_annotations.zip")
        os.system("mv %s %s" % (filename, datafolder))
        os.system("unzip human_annotations.zip")

    unzipped_file = os.path.join(datafolder, "human_annotations/%s_human.json" % (dataset_type))
    with open(unzipped_file, "r") as f:
        original_dataset = json.load(f)

    dataset_flat = []
    summary_keys = ["gpt3", "t0", "brio"]
    for doc_id, doc in original_dataset.items():
        summary_scores = {k: 0 for k in summary_keys}
        for anno in doc["annotators"]:
            summary_scores[anno["best_summary"][0]] += 1
            summary_scores[anno["worst_summary"][0]] -= 1
        
        for k in summary_keys:
            dataset_flat.append({"doc_id": doc_id, "document": doc["article"], "summary": doc[k]["text"], "system": k, "score": summary_scores[k]})
    return generic_nnd_generator(dataset_flat, group_key="doc_id", candidate_key="summary", quality_key="score", document_key="document", candidate_copy_keys=["system"], minimum_gap=min_score_gap)

# TASK 2: QUESTION GENERATION
def mark_paragraph_answer(paragraph, answer, model_card=""):
    if "prophetnet" in model_card:
        return "%s [SEP] %s" % (answer, paragraph)
    elif "mixqg" in model_card:
        return f"{answer} \\n {paragraph}"
    elif "macaw" in model_card:
        return f"$question$ ; $context$ = {paragraph} ; $answer$ = {answer}"
    else:
        return "%s \n %s" % (answer, paragraph) # The default, used for our trained models

def load_qd_nnd(datafolder, model_card="gpt2-medium"):
    qd_groups = []
    filename = "quiz_design_groups.jsonl"
    full_path = os.path.join(datafolder, filename)
    if not os.path.exists(full_path):
        # Download from Github into datafolder
        print("Downloading %s from Github" % (filename))
        os.system("wget https://raw.githubusercontent.com/salesforce/QGen/main/Quiz_Design/quiz_design_groups.jsonl")
        os.system("mv %s %s" % (filename, datafolder))

    with open(full_path, "r") as f:
        for line in f:
            qd_groups.append(json.loads(line))

    flat_questions = []
    for group in qd_groups:
        doc = mark_paragraph_answer(group["context"], group["answer_span"], model_card=model_card)
        for q in group["questions"]:
            q["document"] = doc
            q["group_id"] = group["group_id"]
        flat_questions += group["questions"]
    return generic_nnd_generator(flat_questions, group_key="group_id", candidate_key="question", quality_key="reason", keep_key_as_is=True, document_key="document", common_copy_keys=[], candidate_copy_keys=[], must_include_label="No error")

# TASK 3: GENERATIVE QUESTION ANSWERING
def qa_model_modification(document, model_card):
    if "macaw" in model_card:
        return f"$answer$ ; $question$ = {document}"
    return document

def load_c300_groups(datafolder, model_card):
    filename = "challenge300-outputs.tsv"
    full_path = os.path.join(datafolder, filename)
    if not os.path.exists(full_path):
        print("Downloading %s from Github" % (filename))
        os.system("wget https://raw.githubusercontent.com/allenai/macaw/main/challenge300-outputs.tsv")
        os.system("mv %s %s" % (filename, datafolder))

    label_groups = {"Common Sense": ["commonsense", "explanation", "general knowledge", "human behavior", "hypothetical", "story understanding", "Winograd", "history"],
                    "Entity": ["entity substitution", "entity tracking"],
                    "Comparison": ["comparison", "estimation", "temporal"],
                    "Creativity": ["example generation", "false presupposition", "generation", "riddle"],
                    "Science": ["math", "meta-reasoning", "science", "spatial", "steps"]}
    label2group = {label: group for group, labels in label_groups.items() for label in labels}

    with open(full_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        data = [dict(zip(header, row)) for row in reader]

    model_names = ["Macaw-11B", "Macaw-answer-11B", "GPT3-davinci", "Jurassic-1-jumbo", "T5-XXL-SSM-NQ"]
    dataset_groups = []
    for d in data:
        dataset_groups.append({"id": d["id"], "question": qa_model_modification(d["question"], model_card), "category": d["category"], "cat_group": label2group[d["category"]],
                 "answers": [{"answer": d[mn], "credit": float(d["credit-%s" % (mn)]), "model_name": mn} for mn in model_names]})
    return dataset_groups

def load_c300_nnd(datafolder=".", model_card=""):
    dataset_groups = load_c300_groups(datafolder, model_card)
    flat_answers = []
    for group in dataset_groups:
        static = {k: group[k] for k in ["id", "question", "cat_group", "category"]}
        for answer in group["answers"]:
            answer.update(static)
        flat_answers += [ans for ans in group["answers"] if ans["credit"] in [0.0, 1.0]]
    return generic_nnd_generator(flat_answers, group_key="id", candidate_key="answer", quality_key="credit", keep_key_as_is=True, document_key="question", common_copy_keys=["cat_group"])

# TASK 4: MACHINE TRANSLATION
def load_mt_mqm_nnd(datafolder=".", label_type="category"):
    # Based on this Github repo: https://github.com/google/wmt-mqm-human-evaluation
    # Based on this paper: https://arxiv.org/pdf/2104.14478.pdf
    assert label_type in ["category", "severity"]
    file_name = "mqm_newstest2021_ende.tsv"
    full_path = os.path.join(datafolder, file_name)
    if not os.path.exists(file_name):
        # Download from Github into datafolder
        print("Downloading %s from Github" % (file_name))
        os.system("wget https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/main/newstest2021/ende/%s" % (file_name))
        os.system("mv %s %s" % (file_name, datafolder))

    dataset = []
    with open(full_path, "r", encoding="utf-8") as f:
        headers = f.readline().strip().split("\t")
        for line in f:
            tokens = line.strip().split("\t")
            if len(tokens) != 9:
                continue # Some of the rows have extra tabs, we skip them as they cause problems
            row = dict(zip(headers, tokens))
            row["doc_seg"] = "%d_%d_%s" % (int(row["doc_id"]), int(row["seg_id"]), row["source"])
            row["target"] = row["target"].replace("<v>", "").replace("</v>", "")
            dataset.append(row)
    return generic_nnd_generator(dataset, group_key="doc_seg", candidate_key="target", quality_key=label_type, keep_key_as_is=True, document_key="source", candidate_copy_keys=["system", "severity", "category"], must_include_label="No-error")
