using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

public class AI_NonUnity
{
    public bool go = true;
    public int cIterator = 0;
    public int addition = 0;
    public float adjustedFitness;
    public float speed = 0.1f;
    public int recursionAddLink = 0;
    public float howIsGood = 0;
    public float textHowIsGood = 0;
    private float accumulatedFitness = 0f;
    private int processedInBatch = 0;

    private float accumulatedNewFitness = 0f;
    private int processedNewInBatch = 0;
    public float newPhraseFitness = 0f;

    // Network structure
    public List<float> neurones = new List<float>();
    public List<int> inpInnov = new List<int>();
    public List<int> outInnov = new List<int>();
    public List<float> weights = new List<float>();
    public List<bool> actConnect = new List<bool>();
    public List<bool> RNNs = new List<bool>();
    public List<float> RNNneurones = new List<float>();
    public List<int> order = new List<int>();
    public List<Dictionary<int, float>> adjList = new List<Dictionary<int, float>>();
    public Dictionary<int, bool> neuronHasFFIncoming = new Dictionary<int, bool>();
    public List<int> innovations = new List<int>();

    // Data
    public string inputPhrase;
    public string targetPhrase;
    public string generatedText;
    public bool ifNew = false;

    // Tokenized data
    public List<int> inputTokens = new List<int>();
    public List<int> targetTokens = new List<int>();
    public List<int> outputTokens = new List<int>();

    public int prevNumber = 0;
    public int thisNumber = 0;
    private int outConnections = 1;
    private int initalNeurones;
    public int desiredNeurones;

    public int seedLength = 4;
    private int seedIndex = 0;
    private bool generationComplete = false;
    private List<int> currentContext = new List<int>();
    private bool seedProcessed = false;
    private List<int> seedTokens = new List<int>();
    private int expectedTokenId = -1;

    public Tokenizer tokenizer;
    public int vocabSize;

    private HashSet<int> availableTokens = new HashSet<int>();
    private List<int> availableInputNeurons = new List<int>();
    private List<int> availableOutputNeurons = new List<int>();
    private Random random;
    public GameManager_NonUnity spawnerOfNN;
    public string name;
    private Logger logger;

    public AI_NonUnity(GameManager_NonUnity spawner, Random rnd, Logger log = null)
    {
        spawnerOfNN = spawner;
        random = rnd;
        logger = log;
    }

    public void Start()
    {
        go = true;
        generationComplete = false;
        howIsGood = 0;
        textHowIsGood = 0;
        cIterator = 0;
        generatedText = "";
        outputTokens.Clear();
        ResetBatchStats();

        initalNeurones = 2 + vocabSize + vocabSize;

        if (neurones.Count < initalNeurones)
        {
            neurones.Clear();
            RNNneurones.Clear();

            neurones.Add(1f);
            RNNneurones.Add(1f);
            neurones.Add(0f);
            RNNneurones.Add(0f);

            for (int i = 0; i < vocabSize; i++)
            {
                neurones.Add(0f);
                RNNneurones.Add(0f);
            }

            for (int i = 0; i < vocabSize; i++)
            {
                neurones.Add(0f);
                RNNneurones.Add(0f);
            }
        }
        else
        {
            for (int i = 0; i < neurones.Count; i++)
            {
                neurones[i] = 0f;
                RNNneurones[i] = 0f;
            }
            neurones[0] = 1f;
            RNNneurones[0] = 1f;
        }

        outConnections = 2 + vocabSize;

        for (int i = 0; i < inpInnov.Count; i++)
        {
            if (inpInnov[i] >= neurones.Count || outInnov[i] >= neurones.Count)
            {
                inpInnov.RemoveAt(i);
                outInnov.RemoveAt(i);
                weights.RemoveAt(i);
                actConnect.RemoveAt(i);
                RNNs.RemoveAt(i);
                innovations.RemoveAt(i);
                i--;
            }
        }

        UpdateAvailableNeurons(availableTokens);
        Adder();
        makeOrder();

        if (targetTokens == null || targetTokens.Count < 3)
        {
            logger?.Warning("Target tokens too short, generation disabled");
            generationComplete = true;
            go = false;
        }
    }

    private const float NEW_PHRASE_MULTIPLIER = 1.5f;  // Consistency: must match GetCombinedFitness

    public void ProcessStep()
    {
        int totalTargetLength = targetTokens.Count;
        if (totalTargetLength < 3)
        {
            generationComplete = true;
            go = false;
            return;
        }

        int firstPredictionIndex = seedLength;
        int lastPredictionIndex = Math.Max(seedLength, totalTargetLength - 2);  // Fix: guarantee at least one prediction

        if (firstPredictionIndex < 1) firstPredictionIndex = 1;
        if (firstPredictionIndex > lastPredictionIndex)
        {
            generationComplete = true;
            go = false;
            return;
        }

        List<float> cachedRNNState = null;

        for (int pos = firstPredictionIndex; pos <= lastPredictionIndex; pos++)
        {
            // Safety check: ensure pos + 1 is within bounds for targetTokens
            if (pos + 1 >= targetTokens.Count)
                break;

            if (pos == firstPredictionIndex)
            {
                // First prediction: process all context tokens from scratch
                ResetNeurons();

                for (int ctxIdx = 1; ctxIdx <= pos; ctxIdx++)
                {
                    // Safety: ensure context token exists
                    if (ctxIdx >= targetTokens.Count)
                        break;

                    int token = targetTokens[ctxIdx];

                    for (int j = 2; j < 2 + vocabSize; j++)
                        neurones[j] = 0f;
                    int inputIdx = 2 + token;
                    // Safety: ensure inputIdx is within neuron array bounds
                    if (inputIdx >= 0 && inputIdx < neurones.Count)
                        neurones[inputIdx] = 1f;

                    neurones[1] = 0f;
                    RunNetwork();
                    UpdateRNNState();
                }

                // Cache RNN state before prediction (like KV-cache in transformers)
                cachedRNNState = new List<float>(RNNneurones);
            }
            else
            {
                // Subsequent predictions: restore cached RNN state and process only the new token
                RNNneurones = new List<float>(cachedRNNState);
                // Initialize neurones from RNN state
                for (int i = 0; i < Math.Min(RNNneurones.Count, neurones.Count); i++)
                    neurones[i] = RNNneurones[i];

                // Process only the new token
                int token = targetTokens[pos];

                for (int j = 2; j < 2 + vocabSize; j++)
                    neurones[j] = 0f;
                int inputIdx = 2 + token;
                // Safety: ensure inputIdx is within neuron array bounds
                if (inputIdx >= 0 && inputIdx < neurones.Count)
                    neurones[inputIdx] = 1f;

                neurones[1] = 0f;
                RunNetwork();
                UpdateRNNState();

                // Update cache with new RNN state for next token
                cachedRNNState = new List<float>(RNNneurones);
            }

            neurones[1] = 1f;
            RunNetwork();

            int predictedToken = GetPredictedToken();
            outputTokens.Add(predictedToken);

            int expectedToken = targetTokens[pos + 1];
            float fitness = 0f;

            if (predictedToken == expectedToken)
            {
                fitness = 1f;
            }
            else
            {
                float[] outputActivations = new float[vocabSize];
                for (int i = 0; i < vocabSize; i++)
                {
                    int idx = outConnections + i;
                    outputActivations[i] = neurones[idx];
                }
                float temperature = 0.5f;
                float[] probs = Softmax(outputActivations, temperature);
                if (expectedToken >= 0 && expectedToken < vocabSize)
                    fitness = probs[expectedToken];
            }

            if (ifNew)
            {
                accumulatedNewFitness += fitness * NEW_PHRASE_MULTIPLIER;
                processedNewInBatch += 1;  // Count it as one sample, weighting is in accumulatedNewFitness
            }
            accumulatedFitness += fitness;
            processedInBatch++;
        }

        if (tokenizer != null)
            generatedText = tokenizer.Detokenize(outputTokens);

        generationComplete = true;
        go = false;
    }

    private void ResetNeurons()
    {
        for (int i = 0; i < neurones.Count; i++)
        {
            neurones[i] = 0f;
            RNNneurones[i] = 0f;
        }
        neurones[0] = 1f;
        RNNneurones[0] = 1f;
    }

    private void RunNetwork()
    {
        for (int i = 0; i < order.Count; i++)
        {
            int thisNeuron = order[i];
            if (thisNeuron >= 2 + vocabSize)
            {
                neurones[thisNeuron] = (float)activationFunction(neurones[thisNeuron]);
            }
            if (adjList.Count > thisNeuron)
            {
                foreach (var b in adjList[thisNeuron])
                    if (b.Key < neurones.Count)
                        neurones[b.Key] += b.Value * neurones[thisNeuron];
            }
        }
    }

    private int GetPredictedToken()
    {
        float maxActivation = -999f;
        int predictedToken = Tokenizer.UNK_TOKEN;

        int outputStart = outConnections;
        int outputEnd = Math.Min(outConnections + vocabSize, neurones.Count);

        for (int i = outputStart; i < outputEnd; i++)
        {
            float activation = neurones[i];
            if (activation > maxActivation)
            {
                maxActivation = activation;
                predictedToken = i - outConnections;
            }
        }

        if (maxActivation <= -0.9f || predictedToken < 0 || predictedToken >= vocabSize)
            return Tokenizer.UNK_TOKEN;

        return predictedToken;
    }

    private void UpdateRNNState()
    {
        for (int i = 0; i < inpInnov.Count; i++)
        {
            if (actConnect[i] && RNNs[i])
            {
                if (outInnov[i] < RNNneurones.Count)
                {
                    RNNneurones[outInnov[i]] += neurones[inpInnov[i]] * weights[i];
                }
            }
        }

        for (int i = 0; i < Math.Min(RNNneurones.Count, neurones.Count); i++)
        {
            neurones[i] = RNNneurones[i];
        }

        for (int i = 0; i < RNNneurones.Count; i++)
        {
            RNNneurones[i] *= 0.95f;
        }
    }

    public void SetTokenizer(Tokenizer tokenizer)
    {
        this.tokenizer = tokenizer;
        this.vocabSize = tokenizer.GetVocabSize();
        outConnections = 2 + vocabSize;
    }

    public void UpdateAvailableTokens(HashSet<int> tokens)
    {
        availableTokens = new HashSet<int>(tokens);
        UpdateAvailableNeurons(availableTokens);
    }

    public void AddNode()
    {
        int ind = random.Next(0, outInnov.Count);
        int reccurency = 0;
        while (reccurency < 5)
        {
            if (!RNNs[ind] && actConnect[ind])
            {
                break;
            }
            ind = random.Next(0, outInnov.Count);
            ++reccurency;
        }
        if (reccurency >= 5)
        {
            return;
        }
        neurones.Add(0);
        adjList.Add(new Dictionary<int, float>());

        actConnect[ind] = false;

        weights.Add(weights[ind]);
        inpInnov.Add(inpInnov[ind]);
        outInnov.Add(neurones.Count - 1);
        RNNs.Add(false);
        actConnect.Add(true);
        RNNneurones.Add(0);
        innovations.Add(spawnerOfNN.DealInnovations(inpInnov[inpInnov.Count - 1], outInnov[inpInnov.Count - 1], RNNs[inpInnov.Count - 1]));

        weights.Add(1f);
        inpInnov.Add(neurones.Count - 1);
        outInnov.Add(outInnov[ind]);
        RNNs.Add(false);
        actConnect.Add(true);
        innovations.Add(spawnerOfNN.DealInnovations(inpInnov[inpInnov.Count - 1], outInnov[inpInnov.Count - 1], RNNs[inpInnov.Count - 1]));
        makeOrder();
    }

    public void AddLink()
    {
        bool errorInOut = false;
        weights.Add((float)(random.NextDouble() * 6 - 3));

        if (availableInputNeurons.Count == 0)
        {
            return;
        }
        inpInnov.Add(availableInputNeurons[random.Next(0, availableInputNeurons.Count)]);

        List<int> TakenConnections = new List<int>();

        if (availableOutputNeurons.Count == 0)
        {
            return;
        }
        int probableOut = availableOutputNeurons[random.Next(0, availableOutputNeurons.Count)];

        for (int i = 0; i < outInnov.Count; i++)
        {
            if (inpInnov[i] == inpInnov[inpInnov.Count - 1])
            {
                TakenConnections.Add(outInnov[i]);
            }
        }

        foreach (int a in TakenConnections)
        {
            if (a == probableOut || probableOut == inpInnov.Last())
            {
                errorInOut = true;
                break;
            }
        }

        if (probableOut >= neurones.Count)
        {
            probableOut = neurones.Count - 1;
        }

        outInnov.Add(probableOut);
        RNNs.Add(false);
        actConnect.Add(true);

        if (errorInOut || GenToPh().SequenceEqual(new List<int> { 1, outConnections, 0 }))
        {
            inpInnov.RemoveAt(inpInnov.Count - 1);
            outInnov.RemoveAt(outInnov.Count - 1);
            RNNs.RemoveAt(RNNs.Count - 1);
            actConnect.RemoveAt(actConnect.Count - 1);
            weights.RemoveAt(weights.Count - 1);
            makeOrder();
            ++recursionAddLink;
            if (recursionAddLink < 3)
            {
                AddLink();
                recursionAddLink = 0;
            }
        }
        else
        {
            if (random.Next(0, 6) >= 3)
            {
                if (RNNs.Count > 0)
                {
                    RNNs[RNNs.Count - 1] = true;
                    innovations.Add(spawnerOfNN.DealInnovations(inpInnov[inpInnov.Count - 1], outInnov[inpInnov.Count - 1], RNNs[inpInnov.Count - 1]));
                }
            }
            else
            {
                innovations.Add(spawnerOfNN.DealInnovations(inpInnov[inpInnov.Count - 1], outInnov[inpInnov.Count - 1], RNNs[inpInnov.Count - 1]));
            }
        }
        makeOrder();
    }

    public List<int> GenToPh()
    {
        List<float> _weights = new List<float>(weights);
        List<int> _inpInnov = new List<int>(inpInnov);
        List<int> _outInnov = new List<int>(outInnov);
        List<bool> _actConnect = new List<bool>(actConnect);
        List<bool> _RNNs = new List<bool>(RNNs);

        HashSet<int> usedNeurons = new HashSet<int>();
        usedNeurons.Add(0);

        for (int i = 0; i < _actConnect.Count; i++)
        {
            if (_actConnect[i] && !_RNNs[i])
            {
                usedNeurons.Add(_inpInnov[i]);
                usedNeurons.Add(_outInnov[i]);
            }
        }

        // CRITICAL FIX: Ensure adjList is large enough before using it
        // AddNode() mutations add neurons but adjList was built at different time
        while (adjList.Count < neurones.Count)
        {
            adjList.Add(new Dictionary<int, float>());
        }

        for (int i = 0; i < _actConnect.Count; i++)
        {
            if (_actConnect[i] && !_RNNs[i])
            {
                // Guard against invalid indices
                if (_inpInnov[i] < adjList.Count && _outInnov[i] < neurones.Count)
                {
                    adjList[_inpInnov[i]][_outInnov[i]] = _weights[i];
                }
            }
        }

        Dictionary<int, int> inDegree = new Dictionary<int, int>();
        foreach (int n in usedNeurons)
            inDegree[n] = 0;

        for (int i = 0; i < neurones.Count; i++)
        {
            if (!usedNeurons.Contains(i)) continue;
            foreach (var kv in adjList[i])
            {
                int to = kv.Key;
                if (usedNeurons.Contains(to))
                    inDegree[to]++;
            }
        }

        List<int> nullConn = new List<int>();
        foreach (int n in usedNeurons)
            if (inDegree[n] == 0)
                nullConn.Add(n);

        List<int> order1 = new List<int>();
        for (int i = 0; i < nullConn.Count; i++)
        {
            int u = nullConn[i];
            order1.Add(u);
            foreach (var kv in adjList[u])
            {
                int v = kv.Key;
                if (usedNeurons.Contains(v))
                {
                    inDegree[v]--;
                    if (inDegree[v] == 0)
                        nullConn.Add(v);
                }
            }
        }

        if (order1.Count != usedNeurons.Count)
        {
            return new List<int> { 1, outConnections, 0 };
        }

        return order1;
    }

    public void makeOrder()
    {
        order = GenToPh();
        neuronHasFFIncoming.Clear();
        foreach (int n in order)
            neuronHasFFIncoming[n] = false;
        for (int i = 0; i < inpInnov.Count; i++)
        {
            if (actConnect[i] && !RNNs[i])
            {
                int to = outInnov[i];
                if (neuronHasFFIncoming.ContainsKey(to))
                    neuronHasFFIncoming[to] = true;
            }
        }
    }

    public void Adder()
    {
        adjList.Clear();
        for (int i = 0; i < neurones.Count; i++)
        {
            adjList.Add(new Dictionary<int, float>());
        }
        if (addition == 1)
        {
            AddLink();
        }
        if (addition == 2)
        {
            AddNode();
        }
        makeOrder();
        addition = 0;
    }

    public bool correctGen(List<int> a)
    {
        if (a == null || a.Count == 0) return false;
        if (a.SequenceEqual(new List<int> { 1, outConnections, 0 })) return false;
        return true;
    }

    public void ResetForNewPhase()
    {
        go = true;
        generationComplete = false;
        cIterator = 0;
        generatedText = "";
        outputTokens.Clear();

        for (int i = 0; i < neurones.Count; i++)
        {
            neurones[i] = 0f;
            RNNneurones[i] = 0f;
        }
        neurones[0] = 1f;
        RNNneurones[0] = 1f;

        UpdateAvailableNeurons(availableTokens);

        if (targetTokens.Count > 2)
        {
            int maxSeed = Math.Min(seedLength, targetTokens.Count - 2);
            seedTokens = targetTokens.Skip(1).Take(maxSeed).ToList();
            if (seedTokens.Count == 0)
                seedTokens.Add(Tokenizer.SOS_TOKEN);
        }
        else
        {
            seedTokens = new List<int>() { Tokenizer.SOS_TOKEN };
        }

        seedProcessed = false;
        generationComplete = false;
        seedIndex = 0;
        currentContext.Clear();
        outputTokens.Clear();
    }

    public void UpdateAvailableNeurons(HashSet<int> activeTokens)
    {
        availableInputNeurons.Clear();
        availableOutputNeurons.Clear();

        availableInputNeurons.Add(0);
        availableInputNeurons.Add(1);

        foreach (int token in activeTokens)
        {
            int inputIdx = token + 2;
            if (inputIdx < neurones.Count)
                availableInputNeurons.Add(inputIdx);
        }

        foreach (int token in activeTokens)
        {
            int outputIdx = outConnections + token;
            if (outputIdx < neurones.Count)
                availableOutputNeurons.Add(outputIdx);
        }

        int hiddenStart = outConnections + vocabSize;
        for (int i = hiddenStart; i < neurones.Count; i++)
        {
            availableInputNeurons.Add(i);
            availableOutputNeurons.Add(i);
        }
    }

    public NeatModelData ToModelData()
    {
        var data = new NeatModelData();
        data.vocabulary = tokenizer.ExportVocabulary();
        data.vocabSize = vocabSize;
        data.outputStart = outConnections;
        data.neuronCount = neurones.Count;

        data.order = order.ToArray();
        data.neuronHasFFIncoming = new bool[neurones.Count];
        foreach (var kv in neuronHasFFIncoming)
            data.neuronHasFFIncoming[kv.Key] = kv.Value;

        data.rnnDecay = 0.95f;

        data.connections = new ConnectionGene[inpInnov.Count];
        for (int i = 0; i < inpInnov.Count; i++)
        {
            data.connections[i] = new ConnectionGene
            {
                from = inpInnov[i],
                to = outInnov[i],
                weight = weights[i],
                enabled = actConnect[i],
                recurrent = RNNs[i]
            };
        }

        return data;
    }

    public float GetBatchAverageFitness()
    {
        if (processedInBatch == 0)
        {
            textHowIsGood = 0f;
            return 0f;
        }
        textHowIsGood = accumulatedFitness / processedInBatch;
        return textHowIsGood;
    }

    public float GetCombinedFitness(float newPhraseWeight = 1.5f)
    {
        float oldAvg = processedInBatch > 0 ? accumulatedFitness / processedInBatch : 0f;
        float newAvg = processedNewInBatch > 0 ? accumulatedNewFitness / processedNewInBatch : 0f;
        float totalSteps = processedInBatch + newPhraseWeight * processedNewInBatch;
        if (totalSteps == 0) return 0f;
        return (accumulatedFitness + newPhraseWeight * accumulatedNewFitness) / totalSteps;
    }

    private float[] Softmax(float[] activations, float temperature = 1.0f)
    {
        float[] exp = new float[activations.Length];

        // Compute exponentials with numerical stability
        float maxActivation = activations.Length > 0 ? activations[0] : 0f;
        for (int i = 1; i < activations.Length; i++)
            if (activations[i] > maxActivation)
                maxActivation = activations[i];

        for (int i = 0; i < activations.Length; i++)
        {
            // Subtract max for numerical stability (prevents overflow)
            exp[i] = (float)Math.Exp((activations[i] - maxActivation) / temperature);
        }

        float sum = 0f;
        for (int i = 0; i < exp.Length; i++)
            sum += exp[i];

        // Guard against zero sum (all activations → -∞)
        if (sum == 0f) sum = 1f;

        for (int i = 0; i < exp.Length; i++)
        {
            exp[i] /= sum;
        }

        return exp;
    }

    public double activationFunction(double x)
    {
        return Math.Tanh(x);
    }

    public void ResetBatchStats()
    {
        accumulatedFitness = 0f;
        processedInBatch = 0;
        accumulatedNewFitness = 0f;
        processedNewInBatch = 0;
    }

    public int getInitalNeurones()
    {
        return initalNeurones;
    }

    public int getOutConnections()
    {
        return outConnections;
    }
}
