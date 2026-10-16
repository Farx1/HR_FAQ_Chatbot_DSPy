# Evaluation: definitions and method

This page defines the metrics and intervals used to evaluate the chatbot, and the LoRA update behind the fine-tuned adapter. Notation: $Q$ is the set of test questions, $n = |Q|$.

## Retrieval metrics

For a question $q$, the retriever returns a ranked list of chunks. $\text{rank}_q$ is the position of the first relevant chunk (the gold section), or $\infty$ if it is not returned.

**Recall@k** (one relevant chunk per question, so it equals hit rate@k):

$$\text{Recall@}k = \frac{1}{n} \sum_{q \in Q} \mathbb{1}[\text{rank}_q \le k]$$

**Mean Reciprocal Rank** (Voorhees, TREC-8 QA track), cut at $k$:

$$\text{MRR@}k = \frac{1}{n} \sum_{q \in Q} \frac{\mathbb{1}[\text{rank}_q \le k]}{\text{rank}_q}$$

**nDCG@k** (Järvelin and Kekäläinen, 2002) with graded relevance $rel_i$ at position $i$:

$$\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}, \qquad \text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$

where IDCG@k is the DCG of the ideal ordering. With a single relevant chunk ($rel = 1$), $\text{nDCG@}k = \mathbb{1}[\text{rank}_q \le k] / \log_2(\text{rank}_q + 1)$.

Recall@k says whether the answer reaches the generator at all. MRR and nDCG also reward putting it first, which matters because the API truncates the context to 2,000 characters.

## Confidence intervals

### Proportions: Wilson score interval

For $k$ successes out of $n$, $\hat p = k/n$ and $z = 1.96$ for 95 %:

$$\frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \pm \frac{z}{1 + \frac{z^2}{n}} \sqrt{\frac{\hat p (1 - \hat p)}{n} + \frac{z^2}{4n^2}}$$

Used for Recall@k, OOD rejection rate and template-fallback rate. Unlike the normal approximation $\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$, it stays inside $[0, 1]$ and does not collapse to width 0 when $\hat p \in \{0, 1\}$, which happens often with $n \approx 40$ (Wilson, 1927).

### Means: percentile bootstrap

For a per-question score $s_q$ (ROUGE-L, reciprocal rank), draw $B$ resamples of $Q$ with replacement, compute the mean $\bar s^{*}_b$ of each, and report the 2.5th and 97.5th percentiles of $\{\bar s^{*}_b\}$.

### Comparing two systems: paired bootstrap

Systems $A$ and $B$ answer the same questions. Let $d_q = s_q^B - s_q^A$ and $\delta = \bar d$. Resample the questions (the same indices for both systems), giving $\delta^{*}_b$. The 95 % CI of the difference is the percentile interval of $\{\delta^{*}_b\}$. The one-sided p-value for $H_0: \delta \le 0$ shifts the bootstrap distribution to the null (Berg-Kirkpatrick et al., 2012):

$$p = \frac{1}{B} \sum_{b=1}^{B} \mathbb{1}[\delta^{*}_b - \delta \ge \delta]$$

Pairing removes the variance due to question difficulty, which an unpaired test would count as noise. Dror et al. (2018) recommend the bootstrap or a permutation test for metrics like ROUGE, whose distribution is unknown, rather than a t-test.

For paired binary outcomes (correct / incorrect), the exact McNemar test uses only the discordant pairs: with $b$ items where only $A$ is right and $c$ where only $B$ is right, $p = \min(1, 2 \cdot P[X \le \min(b, c)])$ with $X \sim \text{Binomial}(b + c, 1/2)$.

### What these intervals do not cover

They measure sampling noise over questions from this test set. They do not cover the choice of questions (written by one person from the same documents the retriever indexes), seeds of training, or prompt choices.

## LoRA

LoRA (Hu et al., 2021) freezes a pre-trained weight $W_0 \in \mathbb{R}^{d \times k}$ and learns a low-rank update:

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x, \qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d, k)$$

**Initialization.** $A$ is drawn from a random Gaussian and $B = 0$, so $\Delta W = BA = 0$ at step 0: the adapted model starts with exactly the output of the base model, and training moves away from it gradually.

**Parameter count.** For a square layer ($d = k$), full fine-tuning trains $d^2$ parameters; LoRA trains $dr + rk = 2dr$. The ratio is $2r/d$. For a 768 x 768 projection of DialoGPT-small with $r = 8$, LoRA trains 2.1 % of its parameters; [MODEL_CARD.md](../MODEL_CARD.md) covers the non-square c_attn.

**Scaling.** $\alpha / r$ keeps the update magnitude comparable when $r$ changes; with Adam, tuning $\alpha$ is close to tuning the learning rate.

At inference, $BA$ can be merged into $W_0$, so LoRA adds no latency.

## References

- Berg-Kirkpatrick, T., Burkett, D., Klein, D. (2012). An Empirical Investigation of Statistical Significance in NLP. *EMNLP-CoNLL 2012*, 995-1005. https://aclanthology.org/D12-1091/
- Dror, R., Baumer, G., Shlomov, S., Reichart, R. (2018). The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing. *ACL 2018*, 1383-1392. https://aclanthology.org/P18-1128/
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685. https://arxiv.org/abs/2106.09685
- Järvelin, K., Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM TOIS* 20(4), 422-446. https://doi.org/10.1145/582415.582418
- Voorhees, E. (1999). The TREC-8 Question Answering Track Report. *TREC-8*. https://trec.nist.gov/pubs/trec8/t8_proceedings.html
- Wilson, E. B. (1927). Probable Inference, the Law of Succession, and Statistical Inference. *JASA* 22(158), 209-212. https://doi.org/10.1080/01621459.1927.10502953
