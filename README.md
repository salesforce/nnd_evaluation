# Near-Negative Distinction

Code repository for the paper: [Near-Negative Distinction: Giving a Second Life to Human Evaluation Datasets](https://arxiv.org/abs/2205.06871v1)

<p align="center">
  <img width="350" height="506" src="https://tingofurro.github.io/images/nnd_diagram.png">
</p>

## Motivation

In the NND framework, a generation model is evaluated by measuring whether it passes NND tests. To pass an NND test, the generation model must assigning higher likelihood to high-quality output candidate than to a lower-quality candidate. Candidate quality is based on pre-existing human evaluation datasets.

For example, for the task of generative Question Answering, the Challenge 300 dataset contains the following annotation.
```
Question: ``How could one divert an asteroid heading directly for the Earth?''
Macaw-11b output: ``launch a space shuttle into orbit around it'' -- Credit: 0
Macaw-answer-11b candidate: ``create a spacecraft to intercept and deflect the asteroid'' -- Credit: 1
```

Imagine you want to evaluate a new QA model. The issue is that for this input question, the model might generate a novel answer, say: ``By creating a black hole to absorb the asteroid.''. Because this candidate is not in the human evaluation, it can be challenging to score the candidate (and evaluate the model).

In NND, rather than require the model to generate *new* candidates, we evaluate the likelihood it places on *known* candidates. More specifically, we check whether the model assigns:
```
P(``create a spacecraft to intercept and deflect the asteroid'') > P(``launch a space shuttle into orbit around it'')
```
If it does, we say the model passes the NND test. By creating sets of hundreds of NND tests, we can precisely evaluate text generators.

## Tasks supported

### - Summarization

There are two NND test sets:
- **[Summ Eval NND]** - 3,613 NND tests, measuring model ability on: Consistence, Coherence, Fluency, and Relevance. Based on: https://arxiv.org/abs/2007.12626.
- **[FRANK NND]** - 824 NND tests, focused on factual consistency, more specifically: Semantic Frame, Discourse, and Verifiability errors.  Based on: https://arxiv.org/abs/2104.13346.

See [NND_Summarization.ipynb](https://github.com/MetaMind/nnd/blob/main/NND_Summarization.ipynb) for example use.

### - Question Generation

There is one NND test set:
- **[Quiz Design NND]** - 2686 NND tests, for answer-aware QGen models, measuring ability to avoid: Disfluent, Off Target, and Wrong Context errors. Based on: https://arxiv.org/abs/2205.01730v1

See [NND_QGen.ipynb](https://github.com/MetaMind/nnd/blob/main/NND_QGen.ipynb) for example use.

### - Question Answering

There is one NND test set:
- **[Challenge 300 NND]** - 807 NND tests, for generative QA models, measuring model ability to answer: Common Sense, Comprehension, Entity, Creativity, and Science open-ended questions. Based on: https://arxiv.org/abs/2109.02593

See [NND_QA.ipynb](https://github.com/MetaMind/nnd/blob/main/NND_QA.ipynb) for example use.

### - Machine Translation

There is one NND test set:
- **[WMT 2021 MQM] - 42,675 NND tests, for EN-DE machine translation models, measuring errors along the MQM error categories: Accuracy, Fluency, Terminology, and Locale convention. Also measuring errors based on severity: Minor and Major. Based on: https://arxiv.org/abs/2104.14478

Note: this dataset was not included in the original paper. NND test sets could be expanded to other language pairs (ZH-EN) and years of the MQM competition (WMT 2020) based on the same annotations. See [NND_MT.ipynb](https://github.com/MetaMind/nnd/blob/main/NND_MT.ipynb) for example use.

## Contributing

If you'd like to contribute an NND test set for an existing or a new task, feel free to contribute through Pull Requests, or leave an issue with a public resource you think can be transformed into an NND test set.

If you have questions or suggestions, you can contact us at plaban@salesforce.com. All contributions welcome!
