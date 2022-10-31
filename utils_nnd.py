from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch, os, tqdm, json
from collections import Counter

models_folder = os.environ["MODELS_FOLDER"]

# Unified Generator interface
class GeneratorHF:
    def __init__(self, model_card="gpt2-medium", device="cuda", starter_file=None, max_enc_length=None, max_dec_length=None, force_dec_prepend=None):
        self.model_card = model_card

        self.is_gpt2 = "gpt2" in self.model_card
        if self.is_gpt2:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_card)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_card)
        self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)
        self.max_enc_length = max_enc_length
        self.max_dec_length = max_dec_length
        self.force_dec_prepend = force_dec_prepend

        self.model.eval()

        self.start_id = self.tokenizer.bos_token_id
        self.end_id = self.tokenizer.eos_token_id

        if "prophetnet" in self.model_card:
            # bos_token_id=102, eos_token_id=102
            self.start_id, self.end_id = 102, 102
        if "facebook/wmt19" in self.model_card:
            self.start_id, self.end_id = 2, 2
        if "t5" in self.model_card: # Just for MT
            self.start_id, self.end_id = 0, 1

        if self.start_id is None and self.end_id is not None:
            # For MixQG
            self.start_id = 0

        self.device = device
        if self.is_gpt2:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if starter_file is not None:
            self.reload(starter_file, strict=False)

    def reload(self, from_file, strict=True):
        if not os.path.isfile(from_file):
            # Try to look at the models folder for the file
            from_file = os.path.join(models_folder, from_file)
            assert os.path.isfile(from_file), "Starter file not found, in absolute or in models folder"

        loaded_dict = torch.load(from_file)
        print(self.model.load_state_dict(loaded_dict, strict=strict))

    def save(self, to_file):
        torch.save(self.model.state_dict(), to_file)

    def preprocess(self, encoded_texts, decoded_texts, max_enc_length=None, max_dec_length=None):
        assert len(encoded_texts) == len(decoded_texts), "Mismatch in input/output sizes"

        encoder_ids = self.tokenizer.batch_encode_plus(encoded_texts, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)

        if self.force_dec_prepend is not None:
            decoded_texts = [self.force_dec_prepend + text for text in decoded_texts]
        decoder_tokenized = [self.tokenizer.encode(text=text, add_special_tokens=False) for text in decoded_texts]

        decoder_ids_input = torch.nn.utils.rnn.pad_sequence([torch.LongTensor([self.start_id] + dec) for dec in decoder_tokenized], batch_first=True, padding_value=self.end_id).to(self.device)
        decoder_ids_output = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(dec + [self.end_id]) for dec in decoder_tokenized], batch_first=True, padding_value=-1).to(self.device)

        if self.max_enc_length is not None and max_enc_length is None:
            max_enc_length = self.max_enc_length
        if self.max_dec_length is not None and max_dec_length is None:
            max_dec_length = self.max_dec_length

        if max_enc_length is not None:
            encoder_ids = encoder_ids[:, :max_enc_length]

        if max_dec_length is not None:
            decoder_ids_input = decoder_ids_input[:, :max_dec_length]
            decoder_ids_output = decoder_ids_output[:, :max_dec_length]

        return encoder_ids, decoder_ids_input, decoder_ids_output

    def score_batch(self, encoded_texts, decoded_texts, max_enc_length=None, max_dec_length=None):
        encoder_ids, decoder_ids_input, decoder_ids_output = self.preprocess(encoded_texts, decoded_texts, max_enc_length, max_dec_length)

        if "facebook/wmt19" in self.model_card:
            # There's a problem with this model for now.
            with torch.no_grad():
                model_output = self.model(input_ids=encoder_ids, decoder_input_ids=decoder_ids_input, return_dict=True, labels=decoder_ids_output)
            return {"scores": [-model_output["loss"]]}

        with torch.no_grad():
            crit = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
            if self.is_gpt2:
                encoder_output = self.model(input_ids=encoder_ids, past_key_values=None, return_dict=True)
                past = encoder_output["past_key_values"]
                decoder_output = self.model(input_ids=decoder_ids_input, past_key_values=past, return_dict=True)
                logits = decoder_output["logits"]
            else:
                model_output = self.model(input_ids=encoder_ids, decoder_input_ids=decoder_ids_input, return_dict=True)
                logits = model_output["logits"]
            N, seqlength, vocab_size = logits.shape

            loss_components = crit(logits.view(N*seqlength, vocab_size), decoder_ids_output.contiguous().view(-1)).reshape(N, seqlength)
            num_words = torch.sum(decoder_ids_output != -1, dim=1)
            score_per_item = (- torch.sum(loss_components, dim=1) / num_words).tolist()
        return {"scores": score_per_item}

    def score(self, encoded_texts, decoded_texts, max_enc_length=None, max_dec_length=None, batch_size=32, progress=False):
        N = len(encoded_texts)
        iterator = range(0, N, batch_size)
        if progress and len(iterator) > 1:
            iterator = tqdm.tqdm(iterator)
        scores = []
        for i in iterator:
            batch_encoded_texts = encoded_texts[i:i+batch_size]
            batch_decoded_texts = decoded_texts[i:i+batch_size]
            batch_scores = self.score_batch(batch_encoded_texts, batch_decoded_texts, max_enc_length, max_dec_length)["scores"]
            scores += batch_scores
        return {"scores": scores}

CACHE_LIKELIHOODS = {}
def compute_likelihood(gen, model_name, document, gen_text):
    global CACHE_LIKELIHOODS
    key = (model_name, document, gen_text)
    if key in CACHE_LIKELIHOODS:
        return CACHE_LIKELIHOODS[key]
    else:
        CACHE_LIKELIHOODS[key] = gen.score([document], [gen_text])["scores"][0]
        return CACHE_LIKELIHOODS[key]

def save_nnd_cache(filename):
    global CACHE_LIKELIHOODS
    with open(filename, "w") as f:
        cached_obj = {"___!!!___".join(k): v for k, v in CACHE_LIKELIHOODS.items()}
        json.dump(cached_obj, f)

def load_nnd_cache(filename):
    global CACHE_LIKELIHOODS
    with open(filename, "r") as f:
        cached_obj = json.load(f)
    CACHE_LIKELIHOODS = {tuple(k.split("___!!!___")): v for k, v in cached_obj.items()}
    print("Reloaded %d values" % (len(CACHE_LIKELIHOODS)))


def run_nnd(dataset, model, model_name, no_error_label="NoE", report_type="count", breakdown_key=None, progress=True):
    """
    Run Near Negative Distinction for a model on a dataset.
    """
    assert any(k in report_type for k in  ["count", "accuracy"])

    ite = dataset
    if progress:
        ite = tqdm.tqdm(ite, desc="NND %s" % model_name)

    error_counts = Counter()
    error_breakdown_counts = Counter()
    for d in ite:
        if model_name == "Max Errors":
            error_counts[d["error1"]] += 1
            error_counts[d["error2"]] += 1
        else:
            LP1 = compute_likelihood(model, model_name, d["document"], d["gen1"])
            LP2 = compute_likelihood(model, model_name, d["document"], d["gen2"])

            chosen_label = d["error1"] if LP1 > LP2 else d["error2"]
            error_counts[chosen_label] += 1
            if breakdown_key is not None:
                error_breakdown_counts["%s_%s" % (d[breakdown_key], chosen_label)] += 1

    accuracy = 100.0 * (error_counts[no_error_label] / sum(error_counts.values()))

    D = {"model_name": model_name, "accuracy": accuracy}
    if "count" in report_type:
        D.update({"C_%s" % (k): v for k, v in error_counts.items()})
        if breakdown_key is not None:
            D.update({"C_%s" % (k): v for k, v in error_breakdown_counts.items()})
    if "accuracy" in report_type:
        category_counts = Counter([d["error1"] for d in dataset] + [d["error2"] for d in dataset])
        for k, v in error_counts.items():
            if k == no_error_label:
                D["A_%s" % (k)] = v / category_counts[k]
            else:
                D["A_%s" % (k)] = 1.0 - v / category_counts[k]
        if breakdown_key is not None:
            category_labels = Counter([d[breakdown_key] for d in dataset])
            for cat_label, cat_count in category_labels.items():
                for k in error_counts.keys():
                    if k == no_error_label:
                        D["A_%s" % (cat_label)] = error_breakdown_counts["%s_%s" % (cat_label, k)] / cat_count
    return D
