## mplementation plan
Project structure
Keep the current setup: • Notebook = all orchestration, experiments, plots, analysis, interpretation • models.py = reusable model-related helpers only, especially for Apple Silicon / MPS optimization
What stays in the notebook • data loading • split generation • agreement analysis • embedding generation • weak labeling • learning curves • confusion matrices • comparison tables • conclusions
What goes into models.py • device setup for Apple Silicon / MPS • tokenizer/model loading for classifier models • train/evaluate/predict helpers • metric calculation helpers • confusion matrix helper • consistent result-format helpers
⸻
Data and split strategy
1.1 Agreement-aware dataset design
Use the Financial PhraseBank agreement subsets explicitly.
Clean subset • 100% agreement subset (allagree)
Lower-agreement subset • combine all subsets that are not 100% agreement • use this for ambiguity / robustness analysis
This makes agreement a first-class part of the notebook instead of a side detail.
⸻
1.2 Two fixed test sets
Create and freeze two separate test sets.
Test set A: clean labels • test_allagree • sampled only from 100% agreement data
Test set B: ambiguous labels • test_disagree • sampled only from the non-100%-agreement pool
Purpose: • test_allagree measures performance on the cleanest labels • test_disagree measures robustness when annotation consensus is weaker
⸻
1.3 Clean training logic
Training should use only data from the 100% agreement subset.
From that subset create: • val_allagree • nested hard-label training sets: • train_100 • train_250 • train_500 • train_1000 • one fixed hidden-label pool: • hidden_allagree_pool
Important naming clarification: • this pool is not truly unlabeled in the dataset • it is human-labeled data whose labels are hidden from the weak-labeling pipeline • that allows fair evaluation of weak labels later
That fixes the earlier terminology problem.
⸻
1.4 Important methodological rule
The hidden-label pool must stay fixed across train sizes.
So: • train_100, train_250, train_500, train_1000 all grow hierarchically • hidden_allagree_pool does not shrink as training size increases
Why: • weak labeling should always have access to the same full pool of “available unlabeled data” • this avoids the methodological flaw where larger train sizes accidentally get less unlabeled data
⸻
Metrics and evaluation focus
You want the focus on negative sentiment.
2.1 Primary metric
Use: • F1 for the negative class (F1_negative)
2.2 Secondary metrics
Also report: • negative recall • negative precision • macro-F1 • accuracy
Reason: • accuracy alone can hide weak performance on the negative class • macro-F1 gives overall class balance • negative-class metrics support the business focus
⸻
2.3 Confusion matrices
For every important model/configuration, include: • confusion matrix on test_allagree • confusion matrix on test_disagree
This is required for understanding: • where negative examples are missed • whether ambiguous-label test data causes more confusion between negative/neutral/positive
⸻
Baseline classifier comparison
Use two transformer classifier baselines: • DistilBERT • ModernBERT
These are trained as standard supervised classifiers on the 100%-agreement training sets.
3.1 Baseline comparison stage
First compare both models on the largest hard-labeled training set.
Evaluate on: • validation set • test_allagree • test_disagree
3.2 Model selection rule
Choose the main downstream classifier according to: 1. best negative-class F1 on validation 2. macro-F1 as tiebreaker 3. runtime/stability on Apple Silicon as practical tiebreaker
The selected best classifier is then reused in the semi-supervised section.
⸻
Learning curves
Build learning curves over increasing hard-label training sizes: • 100 • 250 • 500 • 1000
For both classifier baselines: • DistilBERT • ModernBERT
Evaluate each on both test sets.
4.1 Plot content
At minimum plot: • negative-class F1 vs train size • macro-F1 vs train size
For: • test_allagree • test_disagree
This shows: • how performance scales with more hard labels • whether ambiguous data hurts generalization differently
⸻
Inter-annotator-agreement strategy
You asked for a good strategy here. The cleanest strategy is:
5.1 Use only 100%-agreement data for training
This keeps the hard-label signal as reliable as possible.
5.2 Evaluate on both clean and lower-agreement test data
This isolates the effect of annotator disagreement at evaluation time.
5.3 Add a dedicated analysis section
Include a notebook section such as:
Impact of annotator agreement on model performance
Analyze: • performance gap between test_allagree and test_disagree • confusion matrices for both • false negatives on both sets • whether disagreement particularly affects the negative class
This gives a direct answer to the agreement question without overcomplicating training.
⸻
Embedding comparison for weak labeling
You now want: • SBERT • BERT • use this comparison to show why plain BERT embeddings are weaker for this task
That is a good and very interpretable setup.
6.1 Embedding models
Use two embedding pipelines:
Embedding model A • SBERT-style sentence embedding model
Embedding model B • plain BERT embedding extraction • no task-specific sentence-embedding training • this is expected to be weaker for semantic similarity / nearest-neighbor retrieval
⸻
6.2 Pooling rule
Do not use raw CLS token as the main representation.
Instead:
For SBERT • use the sentence embedding produced by the sentence-transformer model
For BERT • use a pooled sentence representation such as mean pooling over token embeddings • not CLS token
This keeps the comparison fair: • sentence-oriented embedding model vs generic BERT embedding model
The point becomes: • even with reasonable pooling, plain BERT is usually less suitable than SBERT for semantic-neighbor weak labeling
⸻
6.3 Embedding quality analysis before weak labeling
Before running k-NN weak labeling, compare embedding quality directly.
Add similarity-based statistics without dimensionality reduction first.
Recommended analyses: • intra-class cosine similarity • inter-class cosine similarity • nearest-neighbor label purity • class-centroid distance • optional pairwise similarity distributions
Optional afterward: • UMAP or 2D visualization as bonus / illustration only
This fixes the earlier weakness where the notebook jumped too quickly to dimensionality reduction.
⸻
Weak-labeling design
Keep weak labeling focused and interpretable.
7.1 Seed data
Weak-label generation uses only: • hard-labeled seed examples from the 100%-agreement training split
7.2 Target pool
Weak labels are generated for: • hidden_allagree_pool
Again: • these labels are hidden during generation • but available later for evaluation
⸻
7.3 Weak-labeling algorithm
Use k-NN as the main weak-labeling strategy.
But compare multiple parameter settings instead of one fixed setting.
At minimum compare: • k = 3 • k = 5 • k = 11
Optional: • uniform weights • distance weights
This gives a real parameter comparison while keeping the structure simple.
⸻
7.4 Crossed embedding × k-NN setup
Weak-label experiments should compare: • SBERT + kNN(k=3,5,11) • BERT + kNN(k=3,5,11)
This lets you answer: • does SBERT produce better neighborhoods for weak labeling? • how sensitive are results to k? • does plain BERT show the expected weakness?
⸻
7.5 Direct weak-label evaluation
Before retraining any classifier, directly evaluate the weak labels against the hidden true labels of hidden_allagree_pool.
For each embedding + k combination report: • negative-class F1 • macro-F1 • accuracy • confusion matrix
This separates: • embedding/weak-label quality from • downstream classifier benefits
⸻
Semi-supervised retraining
For each train size: 1. take hard-labeled train_n 2. generate weak labels on the fixed hidden_allagree_pool 3. combine hard labels + weak labels 4. retrain the selected best classifier 5. evaluate on both test sets
Do this for the most promising weak-label setups, not necessarily every single one if runtime becomes excessive.
⸻
8.1 Comparisons to include
For each train size compare: • hard-label baseline • direct weak-label quality • semi-supervised retrained classifier
This should make clear: • weak labels may be noisy on their own • but can still improve the downstream classifier
⸻
Time-savings factor
Add a dedicated section to estimate annotation savings.
9.1 Core idea
Ask:
How many manually labeled training examples would be needed in the baseline to reach the same negative-class F1 as the semi-supervised approach?
9.2 Example logic
If: • baseline with 500 hard labels reaches negative F1 = X • semi-supervised setup with 250 hard labels + weak labels also reaches X
Then: • the approach saves about 250 manual labels • equivalent to 50% fewer manually labeled examples
This gives the practical payoff of weak labeling.
⸻
Bonus section cleanup
The previous broken LLM bonus code should not remain executable if it breaks the notebook.
For now: • remove or disable broken bonus code • keep only a markdown note or TODO • ensure notebook runs cleanly from top to bottom
That matters more than keeping incomplete bonus experiments alive.
⸻
Recommended notebook section order
Introduction and task framing
Apple Silicon / MPS setup
Data loading and agreement-aware split generation
Data inspection and class distributions
Definition of two test sets
Baseline classifiers: DistilBERT vs ModernBERT
Baseline learning curves
Agreement impact analysis
Embedding models: SBERT vs BERT
Embedding quality statistics
Weak labeling with k-NN variants
Direct weak-label evaluation
Semi-supervised retraining
Baseline vs semi-supervised comparison
Time-savings factor
Optional bonus / future work
Conclusion
AI usage disclosure