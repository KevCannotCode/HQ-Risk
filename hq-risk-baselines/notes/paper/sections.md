# Paper sections written from this harness

Source text for `scripts/fill_paper_docx.py`. Each `## <heading>` is pasted under the matching heading of the
Word draft; `## Comment: <text>` becomes a Word comment anchored on the paragraph that reads `<text>`;
`## Cell: <table caption> | <row label> | <column header>` replaces one table cell.
Blank lines separate paragraphs; a paragraph in [brackets] is set in italics. Numbers come from `results/tables/table_24_compact.csv`.

## Setup

We evaluate the attacks on three tabular datasets shipped with scikit-learn: Breast Cancer (569 rows, 30 features, 2 classes), Digits (1,797 rows, 64 features, 10 classes) and Wine (178 rows, 13 features, 3 classes). Every run uses a stratified 80/20 split and standardises the features on the training rows only. The test rows are fingerprinted before each run and checked after it, so an attack can only act on the training data, the model or its execution.

Three pipelines cover the layers of Table I. The classical pipeline (layers D and M) trains a logistic regression on the central training set. The federated pipeline (layer F) spreads the training set over 10 clients and trains the same model with FedAvg for 30 rounds of one local epoch, once with an IID partition and once with a non-IID partition drawn from a Dirichlet distribution (α = 0.5). The quantum pipeline (layer Q) trains QuantumNet, an 8-qubit variational circuit with angle encoding and one data re-upload, on the first 8 principal components. Its weights are fitted with Adam on an exact statevector simulation, and the trained circuit is evaluated on the Qiskit Aer sampler with 256 shots, so shot noise is part of every quantum result.

Each attack is a sweep along one intensity axis. On the data layer we flip 5–40 % of the training labels at random, or relabel the same share of source-class rows as a target class (targeted poisoning). On the model layer we run white-box FGSM evasion (ε up to one standard deviation), a backdoor trigger planted in 1–20 % of the training rows, label-only model extraction with up to 1,000 queries, confidence-threshold membership inference, and black-box model inversion. On the federated layer 10–50 % of the clients are malicious and either flip their labels, negate their update, relabel one class, or plant the backdoor. On the quantum layer we inject 1–16 RX(π/2) gates into the trained circuit, forge 5–40 % of the measured shots, raise the depolarizing and readout error up to p = 0.1, and compile the circuit through a malicious transpiler pass that shifts every RZ angle or swaps qubits before readout. The ARF attacks (backdoor, extraction, membership inference, inversion) run against both the classical and the quantum model; the QTF attacks run against the quantum model.

Every configuration runs with 5 seeds, and each run trains its own clean model on the same split, so each clean-versus-attacked comparison is paired. We report test accuracy, the accuracy drop in percentage points, and an attack success rate: the share of correct predictions turned wrong for evasion and the quantum attacks, the share of source or triggered rows sent to the target class for targeted and backdoor attacks, the fidelity of the stolen copy for extraction, the membership accuracy for membership inference, and the cosine similarity to the true class mean for inversion. The matrix has 20 sweeps, 3 datasets and 2,130 runs, executed on one CPU with Python 3.12, scikit-learn 1.9, Qiskit 2.5 and Qiskit Aer 0.17. The code, the configuration files and every result row are available at https://github.com/KevCannotCode/HQ-Risk, with an interactive view at https://hq-risk-explorer.netlify.app.

## Results

Table II gives one number per attack and dataset at the strongest point of each sweep. Four observations feed the score. First, attacks on the model at inference time cause the largest damage: FGSM evasion turns 94–100 % of correct predictions wrong on every dataset, and on the quantum model 16 injected gates, 4 qubit swaps or a 0.8 rad transpiler drift turn 43–90 % of them wrong. Second, the same ARF attack behaves differently on the quantum model. The backdoor reaches 79–88 % success on QuantumNet against 94–98 % on logistic regression, and extraction recovers a copy with 60–94 % fidelity against 98–100 %, because every answer the attacker receives is a 256-shot estimate. Third, the privacy attacks give little signal on these datasets: membership inference stays at chance (43–52 %) on every model, and inversion recovers the class means with a cosine similarity of 0.53–0.88. Finally, in federated learning the sign-flip client is the most damaging behaviour, with a 70–87 pp drop at 50 % malicious clients, while the backdoor client keeps the global accuracy within 8 pp of clean and still reaches 92–98 % success. This last case is the one the weakest-link rule is designed to catch, since the average accuracy hides it.

## Validation protocol

[Draft by Kevin, an attempt to help us write more. Replace or cut freely.]

Before any number enters the score, every run has to be reproducible and every clean-versus-attacked pair has to differ only by the attack. Each run trains its own clean model on the same split and seed as its attacked model, so the drop is paired. Rerunning a configuration gives a byte-identical result row, the test set is fingerprinted before and after each run, and the torch statevector simulator that trains QuantumNet was checked against Qiskit's Statevector to 1e-15 before the quantum sweeps ran. A single check script confirms that the summary, the tables and the public site all still match the raw runs.

We propose to read the empirical inputs of Section IV from Table II at the strongest point of each sweep. The success rate of an attack, scaled to [0, 1], gives the probability P_i of its ARF term and the CTV, TSV, MSV and NSV components of QTF; where success is undefined, the accuracy drop divided by the clean accuracy takes its place. The layers that were scored and not executed (I, C and G) keep their assessed values. The score is then computed per dataset, so that the three datasets act as three independent systems and the risk band can be compared across them.

## Discussion and Limitations

[Draft by Kevin, an attempt to help us write more. Replace or cut freely.]

The results support the weakest-link rule. The averages in Table II hide the attacks that matter most: the backdoor client costs less than 8 pp of global accuracy yet reaches 92–98 % success, and the quantum transpiler pass changes nothing visible in the circuit the user submitted yet turns 43–80 % of correct predictions wrong. A score that averages layers would rate both systems as healthy.

Several limits apply. Every quantum result comes from the Aer simulator with 256 shots; the attacks were not run on hardware, and real two-qubit error rates are higher than the uniform p we used. The datasets are small tabular benchmarks and the classical model is a logistic regression, so a larger model may react differently to the same attacks. The attack definitions, intensity axes and the 50 % malicious-client ceiling are our own choices, and the success rate is undefined for symmetric poisoning and for the label-flip and sign-flip clients. Membership inference stayed at chance on every model, so the privacy term of ARF receives little signal from these datasets. Finally, the layer weights w, β, α, λ and μ are set as flat values in this paper; whether poisoning and membership inference deserve the same weight is a question we leave to the journal version.

## Conclusion and Future Direction

[Draft by Kevin, an attempt to help us write more. Replace or cut freely.]

HQ-Risk gives a hybrid AI-quantum pipeline one score in [0, 1] built from seven layers, an adversarial-AI factor and a quantum-threat factor, and lets the most exposed layer drive the result. We backed the model with 2,130 reproducible runs covering 20 attacks on classical, federated and quantum models over three datasets, and the strongest attack of every sweep is available as one table and one public site. The next steps are to run the quantum attacks on IBM hardware, to calibrate the weights instead of fixing them, to add larger models and defenses, and to apply the full score to a case study that was not built by us.

## Cell: Attack Surface Per Layer | Model (M) | Executed in PoC

FGSM evasion, backdoor, model extraction, membership inference, model inversion

## Cell: Attack Surface Per Layer | Federated (F) | Executed in PoC

Sign-flip, label-flip, targeted flip, backdoor client (IID/non-IID)
