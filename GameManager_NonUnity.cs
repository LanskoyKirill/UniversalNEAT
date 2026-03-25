using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

public class GameManager_NonUnity
{
    private Logger logger;
    private List<AI_NonUnity> AIs = new List<AI_NonUnity>();
    private List<List<AI_NonUnity>> classAIs = new List<List<AI_NonUnity>>();

    private Tokenizer tokenizer;
    private int vocabSize;
    private HashSet<int> availableTokens = new HashSet<int>();

    public int population;
    public int selection;
    private float c1 = 1f;
    private float c2 = 1f;
    private float c3 = 0.6f;
    private float compatibilityThreshold = 15.0f;
    private float baseCompatibilityThreshold = 15.0f;  // Base value before scaling

    public int epoch = 0;
    private int countBatch = 0;
    public int batchSize = 1;

    private List<string> wordTranslate;
    public string englishPhrase;
    public string targetPhrase;

    private Phrases phrasesComponent;

    public List<int> oInn = new List<int>();
    public List<int> iInn = new List<int>();
    public List<bool> RNN = new List<bool>();
    public List<float> numberClassAIs = new List<float>();

    private int currentPhase = 1;
    private int phrasesPerPhase = 1;
    public int initialPhrases = 1;
    public int currentSeedLength = 4;
    public int phrasesIncrement = 1;
    private float phaseThreshold = 0.65f;  // Lowered from 0.85 - previous max was 0.6507, unreachable threshold blocked curriculum
    public int amountOfRightAnswers = 0;

    private Random random;

    public GameManager_NonUnity(Logger log, int populationSize, int selectionSize, int initialPhrasesCount, Random rnd)
    {
        logger = log;
        population = populationSize;
        selection = selectionSize;
        initialPhrases = initialPhrasesCount;
        random = rnd;
    }

    public void SetPhrases(Phrases phrases)
    {
        phrasesComponent = phrases;
    }

    public void SetTokenizer(Tokenizer tok)
    {
        tokenizer = tok;
        vocabSize = tokenizer.GetVocabSize();
    }

    public void Initialize()
    {
        var allPhrases = phrasesComponent.GetAllPhrases();

        if (allPhrases.Count == 0)
        {
            logger.Error("No phrases loaded!");
            return;
        }

        InitializeAvailableTokens();

        wordTranslate = phrasesComponent.GetPhrase(true);
        if (wordTranslate.Count >= 2)
        {
            englishPhrase = wordTranslate[1];
            targetPhrase = wordTranslate[1];
        }

        bool ifNew = wordTranslate.Count >= 3 && wordTranslate[2] == "Yes";

        // Create initial population
        for (int i = 0; i < population; i++)
        {
            AI_NonUnity ai = new AI_NonUnity(this, random, logger);
            ai.SetTokenizer(tokenizer);
            ai.UpdateAvailableTokens(availableTokens);

            ai.inputPhrase = englishPhrase;
            ai.targetPhrase = targetPhrase;
            ai.ifNew = ifNew;
            ai.inputTokens = tokenizer.Tokenize(englishPhrase);
            ai.targetTokens = tokenizer.Tokenize(targetPhrase);
            ai.addition = 1;
            ai.name = random.Next(1, 10000).ToString();
            ai.Start();  // Initialize neural network structure

            AIs.Add(ai);
        }

        logger.Info($"Population of {population} agents initialized");
    }

    public void ProcessEpoch(int epochNum)
    {
        epoch = epochNum;

        // Update compatibility threshold dynamically based on population and genome complexity
        compatibilityThreshold = ComputeCompatibilityThreshold();

        // Wait for all agents to complete generation
        bool allComplete = false;
        while (!allComplete)
        {
            allComplete = true;
            object allCompleteLock = new object();

            // Parallel evaluation of ProcessStep for all agents
            Parallel.ForEach(AIs, ai => {
                if (ai.go)
                {
                    ai.ProcessStep();
                    if (ai.go)
                    {
                        lock (allCompleteLock)
                        {
                            allComplete = false;
                        }
                    }
                }
            });
        }

        // Get next phrase set
        Phrases phrasesComp = phrasesComponent;
        wordTranslate = phrasesComp.GetPhrase(countBatch == 0);

        if (wordTranslate.Count >= 2)
        {
            englishPhrase = wordTranslate[1];
            targetPhrase = wordTranslate[1];
        }
        bool ifNew = wordTranslate.Count >= 3 && wordTranslate[2] == "Yes";

        int targetLen = tokenizer.Tokenize(targetPhrase).Count;
        int maxSeed = Math.Max(2, targetLen - 4);
        if (maxSeed > 0)
        {
            currentSeedLength = random.Next(2, 4);
        }
        else
        {
            currentSeedLength = 0;
        }

        if (countBatch < batchSize)
        {
            List<AI_NonUnity> listOfAIs = AIs.OrderByDescending(o => o.howIsGood).ToList();
            AI_NonUnity _bestAI = listOfAIs[0];
            float newFitness = _bestAI.GetBatchAverageFitness();

            if (_bestAI.ifNew && newFitness >= phaseThreshold)
            {
                ++amountOfRightAnswers;
                // Advance curriculum when threshold is met
                if (phrasesComponent.lengthOfknown < phrasesComponent.GetAllPhrases().Count)
                {
                    phrasesComponent.lengthOfknown++;
                    logger.Info($"Curriculum advanced: lengthOfknown now {phrasesComponent.lengthOfknown}");
                    // CRITICAL: Expand available tokens and update network neurons for new curriculum phase
                    ExpandAvailableTokens();
                    UpdateAllAgentsNeurons();
                }
            }

            foreach (var ai in listOfAIs)
            {
                ai.inputPhrase = englishPhrase;
                ai.targetPhrase = targetPhrase;
                ai.ifNew = ifNew;
                ai.inputTokens = tokenizer.Tokenize(englishPhrase);
                ai.targetTokens = tokenizer.Tokenize(targetPhrase);
                ai.ResetForNewPhase();
                // CRITICAL: Clear accumulated fitness from batch phase before returning
                ai.ResetBatchStats();
            }

            ++countBatch;
            return;
        }

        countBatch = 0;

        // Calculate fitness for all agents
        Parallel.ForEach(AIs, ai => {
            ai.howIsGood = ai.GetCombinedFitness(1.5f);
            ai.ResetBatchStats();
        });

        // ===== STAGE 1: Collect fitness and report best individual =====
        var aisWithFitness = AIs.Select(o => new { AI = o }).ToList();
        var sortedByRaw = aisWithFitness.OrderByDescending(x => x.AI.howIsGood).ToList();

        AI_NonUnity bestAI = sortedByRaw[0].AI;
        string generatedText = bestAI.generatedText;
        if (string.IsNullOrEmpty(generatedText) && bestAI.outputTokens.Count > 0)
            generatedText = tokenizer.Detokenize(bestAI.outputTokens);

        logger.Info($"Epoch {epoch}: Best fitness={bestAI.howIsGood:F4} | Generated: \"{generatedText}\"");

        if (bestAI.ifNew && bestAI.howIsGood >= phaseThreshold)
        {
            amountOfRightAnswers++;
            // CRITICAL: Advance curriculum and update available neurons when counter phase threshold met
            if (phrasesComponent.lengthOfknown < phrasesComponent.GetAllPhrases().Count)
            {
                phrasesComponent.lengthOfknown++;
                logger.Info($"Curriculum advanced (counter phase): lengthOfknown now {phrasesComponent.lengthOfknown}");
                ExpandAvailableTokens();
                UpdateAllAgentsNeurons();
            }
        }

        // ===== STAGE 2: Species classification =====
        classAIs.Clear();
        foreach (AI_NonUnity aiObj in AIs)
        {
            bool added = false;
            foreach (var species in classAIs)
            {
                AI_NonUnity repAI = species[0];
                float dist = CalculateDistance(aiObj, repAI);
                if (dist <= compatibilityThreshold)
                {
                    species.Add(aiObj);
                    added = true;
                    break;
                }
            }
            if (!added)
                classAIs.Add(new List<AI_NonUnity>() { aiObj });
        }

        logger.Info($"Epoch {epoch}: {classAIs.Count} species, population structure: {string.Join(",", classAIs.Select(s => s.Count.ToString()))} agents per species");

        // ===== STAGE 3: Fitness sharing =====
        foreach (var species in classAIs)
        {
            int size = species.Count;
            if (size == 0) continue;

            species.Sort((a, b) => b.howIsGood.CompareTo(a.howIsGood));

            float[,] distances = new float[size, size];

            // Parallel computation: each thread computes distances for its row
            // No lock needed because each thread only writes to row i (unique per thread)
            Parallel.For(0, size, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, i => {
                for (int j = i + 1; j < size; j++)
                {
                    float d = CalculateDistance(species[i], species[j]);
                    distances[i, j] = d;
                    distances[j, i] = d;
                }
            });

            for (int i = 0; i < size; i++)
            {
                AI_NonUnity ai = species[i];
                float sumSh = 0f;
                for (int j = 0; j < size; j++)
                {
                    if (i == j)
                        sumSh += 1f;
                    else
                    {
                        float d = distances[i, j];
                        float sh = (d < compatibilityThreshold) ? (1f - d / compatibilityThreshold) : 0f;
                        sumSh += sh;
                    }
                }
                ai.adjustedFitness = ai.howIsGood / sumSh;
            }
        }

        // ===== STAGE 4: Parent selection with species protection =====
        List<AI_NonUnity> selectedParents = new List<AI_NonUnity>();

        foreach (var species in classAIs)
        {
            if (species.Count > 0)
                selectedParents.Add(species[0]);
        }

        List<AI_NonUnity> remainingCandidates = AIs.Except(selectedParents).ToList();
        remainingCandidates = remainingCandidates.OrderByDescending(x => x.adjustedFitness).ToList();

        int parentsNeeded = selection - selectedParents.Count;
        if (parentsNeeded > 0)
        {
            selectedParents.AddRange(remainingCandidates.Take(parentsNeeded));
        }
        else if (parentsNeeded < 0)
        {
            selectedParents = selectedParents.Take(selection).ToList();
        }

        selectedParents = selectedParents.Take(selection).ToList();

        // ===== STAGE 5: Create offspring =====
        List<AI_NonUnity> NewAIs = new List<AI_NonUnity>();

        for (int i = 0; i < population - selection; i++)
        {
            int speciesIdx = random.Next(0, classAIs.Count);
            var species = classAIs[speciesIdx];
            if (species.Count == 0) continue;

            AI_NonUnity parent1 = species[random.Next(0, species.Count)];
            AI_NonUnity parent2 = species[random.Next(0, species.Count)];

            if (parent2.howIsGood > parent1.howIsGood)
            {
                AI_NonUnity temp = parent1;
                parent1 = parent2;
                parent2 = temp;
            }

            // Create offspring by cloning parent1
            AI_NonUnity offspring = new AI_NonUnity(this, random, logger);
            offspring.SetTokenizer(tokenizer);
            offspring.UpdateAvailableTokens(availableTokens);
            offspring.seedLength = currentSeedLength;
            offspring.inputPhrase = englishPhrase;
            offspring.targetPhrase = targetPhrase;
            offspring.ifNew = ifNew;
            offspring.inputTokens = tokenizer.Tokenize(englishPhrase);
            offspring.targetTokens = tokenizer.Tokenize(targetPhrase);

            // Copy network structure from parent1
            offspring.neurones = new List<float>(parent1.neurones);
            offspring.inpInnov = new List<int>(parent1.inpInnov);
            offspring.outInnov = new List<int>(parent1.outInnov);
            offspring.weights = new List<float>(parent1.weights);
            offspring.actConnect = new List<bool>(parent1.actConnect);
            offspring.RNNs = new List<bool>(parent1.RNNs);
            offspring.RNNneurones = new List<float>(parent1.RNNneurones);
            offspring.innovations = new List<int>(parent1.innovations);

            // Crossover if parents differ
            if (parent1 != parent2)
            {
                Dictionary<int, int> idxMap1 = new Dictionary<int, int>();
                for (int j = 0; j < parent1.innovations.Count; j++)
                    idxMap1[parent1.innovations[j]] = j;

                Dictionary<int, int> idxMap2 = new Dictionary<int, int>();
                for (int j = 0; j < parent2.innovations.Count; j++)
                    idxMap2[parent2.innovations[j]] = j;

                HashSet<int> allInnov = new HashSet<int>(parent1.innovations);
                allInnov.UnionWith(parent2.innovations);

                List<int> newInp = new List<int>();
                List<int> newOut = new List<int>();
                List<float> newWeights = new List<float>();
                List<bool> newEnabled = new List<bool>();
                List<bool> newRNNs = new List<bool>();
                List<int> newInnov = new List<int>();

                foreach (int innov in allInnov)
                {
                    bool in1 = idxMap1.ContainsKey(innov);
                    bool in2 = idxMap2.ContainsKey(innov);

                    if (in1 && in2)
                    {
                        bool takeFrom1 = random.Next(0, 2) == 0;
                        AI_NonUnity source = takeFrom1 ? parent1 : parent2;
                        int srcIdx = takeFrom1 ? idxMap1[innov] : idxMap2[innov];

                        newInp.Add(source.inpInnov[srcIdx]);
                        newOut.Add(source.outInnov[srcIdx]);
                        newWeights.Add(source.weights[srcIdx]);
                        newEnabled.Add(source.actConnect[srcIdx]);
                        newRNNs.Add(source.RNNs[srcIdx]);
                        newInnov.Add(innov);
                    }
                    else
                    {
                        AI_NonUnity source;
                        int srcIdx;
                        if (in1)
                        {
                            source = parent1;
                            srcIdx = idxMap1[innov];
                        }
                        else
                        {
                            source = parent2;
                            srcIdx = idxMap2[innov];
                        }
                        newInp.Add(source.inpInnov[srcIdx]);
                        newOut.Add(source.outInnov[srcIdx]);
                        newWeights.Add(source.weights[srcIdx]);
                        newEnabled.Add(source.actConnect[srcIdx]);
                        newRNNs.Add(source.RNNs[srcIdx]);
                        newInnov.Add(innov);
                    }
                }

                offspring.inpInnov = newInp;
                offspring.outInnov = newOut;
                offspring.weights = newWeights;
                offspring.actConnect = newEnabled;
                offspring.RNNs = newRNNs;
                offspring.innovations = newInnov;
            }

            // Expand neuron arrays if needed
            List<float> initialNeurons = new List<float>(offspring.neurones);
            for (int j = 0; j < offspring.inpInnov.Count; j++)
            {
                if (offspring.inpInnov[j] + 1 > offspring.neurones.Count)
                {
                    int needed = offspring.inpInnov[j] + 1 - offspring.neurones.Count;
                    for (int k = 0; k < needed; k++)
                    {
                        offspring.neurones.Add(0f);
                        offspring.RNNneurones.Add(0f);
                    }
                }
                if (offspring.outInnov[j] + 1 > offspring.neurones.Count)
                {
                    int needed = offspring.outInnov[j] + 1 - offspring.neurones.Count;
                    for (int k = 0; k < needed; k++)
                    {
                        offspring.neurones.Add(0f);
                        offspring.RNNneurones.Add(0f);
                    }
                }
            }

            // Mutations
            int chooseAddition = random.Next(0, 10);
            if (chooseAddition < 3)
                offspring.addition = 1;
            else if (chooseAddition < 6)
                offspring.addition = 2;
            else
                offspring.addition = 0;

            if (offspring.actConnect.Count > 0)
            {
                if (random.Next(0, 10) < 2)
                {
                    List<int> enabledIndices = new List<int>();
                    for (int j = 0; j < offspring.actConnect.Count; j++)
                        if (offspring.actConnect[j])
                            enabledIndices.Add(j);
                    if (enabledIndices.Count > 0)
                    {
                        int idx = enabledIndices[random.Next(0, enabledIndices.Count)];
                        offspring.actConnect[idx] = false;
                    }
                }

                if (random.Next(0, 10) < 1)
                {
                    List<int> disabledIndices = new List<int>();
                    for (int j = 0; j < offspring.actConnect.Count; j++)
                        if (!offspring.actConnect[j])
                            disabledIndices.Add(j);
                    if (disabledIndices.Count > 0)
                    {
                        int idx = disabledIndices[random.Next(0, disabledIndices.Count)];
                        bool wasRNN = offspring.RNNs[idx];
                        offspring.actConnect[idx] = true;
                        if (!wasRNN && !offspring.correctGen(offspring.GenToPh()))
                        {
                            offspring.actConnect[idx] = false;
                        }
                    }
                }
            }

            for (int j = 0; j < offspring.weights.Count; j++)
            {
                float r = (float)random.NextDouble();
                if (r < 0.1f)
                    offspring.weights[j] = (float)(random.NextDouble() * 6 - 3);
                else if (r < 0.3f)
                    offspring.weights[j] += (float)(random.NextDouble() * 1 - 0.5f);
            }

            offspring.Start();
            NewAIs.Add(offspring);
        }

        // ===== STAGE 6: Replace generation =====
        wordTranslate = phrasesComp.GetPhrase(countBatch == 0);

        List<AI_NonUnity> newPopulation = new List<AI_NonUnity>();

        foreach (var survivor in selectedParents)
        {
            newPopulation.Add(survivor);
            survivor.seedLength = currentSeedLength;
            survivor.inputPhrase = englishPhrase;
            survivor.targetPhrase = targetPhrase;
            survivor.ifNew = ifNew;
            survivor.inputTokens = tokenizer.Tokenize(englishPhrase);
            survivor.targetTokens = tokenizer.Tokenize(targetPhrase);
            survivor.ResetForNewPhase();
        }

        newPopulation.AddRange(NewAIs);

        AIs = newPopulation;
        ++epoch;
    }

    private void InitializeAvailableTokens()
    {
        availableTokens.Clear();
        availableTokens.Add(Tokenizer.PAD_TOKEN);
        availableTokens.Add(Tokenizer.SOS_TOKEN);
        availableTokens.Add(Tokenizer.EOS_TOKEN);
        availableTokens.Add(Tokenizer.UNK_TOKEN);

        var allPhrases = phrasesComponent.GetAllPhrases();
        int phrasesToUse = Math.Min(initialPhrases, allPhrases.Count);

        for (int i = 0; i < phrasesToUse; i++)
        {
            var tokens = tokenizer.Tokenize(allPhrases[i][1]);
            foreach (var token in tokens)
            {
                if (token >= 4)
                    availableTokens.Add(token);
            }
        }

        phrasesComponent.lengthOfknown = phrasesToUse;
        currentPhase = phrasesToUse;
    }

    private void ExpandAvailableTokens()
    {
        availableTokens.Clear();
        availableTokens.Add(Tokenizer.PAD_TOKEN);
        availableTokens.Add(Tokenizer.SOS_TOKEN);
        availableTokens.Add(Tokenizer.EOS_TOKEN);
        availableTokens.Add(Tokenizer.UNK_TOKEN);

        var allPhrases = phrasesComponent.GetAllPhrases();
        int phrasesToUse = phrasesComponent.lengthOfknown;

        for (int i = 0; i < phrasesToUse && i < allPhrases.Count; i++)
        {
            var tokens = tokenizer.Tokenize(allPhrases[i][1]);
            foreach (var token in tokens)
            {
                if (token >= 4)
                    availableTokens.Add(token);
            }
        }

        logger.Debug($"Available tokens expanded to {availableTokens.Count} tokens for {phrasesToUse} phrases");
    }

    private void UpdateAllAgentsNeurons()
    {
        Parallel.ForEach(AIs, ai => {
            ai.UpdateAvailableTokens(availableTokens);
        });
    }

    public int DealInnovations(int inoV, int outV, bool rnnV)
    {
        for (int i = 0; i < iInn.Count; i++)
        {
            if (iInn[i] == inoV && oInn[i] == outV && RNN[i] == rnnV)
                return i;
        }

        iInn.Add(inoV);
        oInn.Add(outV);
        RNN.Add(rnnV);
        return iInn.Count - 1;
    }

    public void ExportBestModel(string filePath)
    {
        var sortedAIs = AIs.OrderByDescending(x => x.howIsGood).ToList();
        if (sortedAIs.Count == 0)
        {
            logger.Error("No AI found to export.");
            return;
        }

        AI_NonUnity best = sortedAIs[0];
        best.makeOrder();
        NeatModelData modelData = best.ToModelData();

        try
        {
            string directory = Path.GetDirectoryName(filePath);
            if (!Directory.Exists(directory))
                Directory.CreateDirectory(directory);

            string json = SerializeToJson(modelData);
            File.WriteAllText(filePath, json);
            logger.Info($"Model exported to {filePath}");
        }
        catch (Exception ex)
        {
            logger.Error($"Failed to export model: {ex.Message}");
        }
    }

    private string SerializeToJson(NeatModelData data)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("{");
        sb.AppendLine("  \"vocabulary\": [");
        if (data.vocabulary != null)
        {
            for (int i = 0; i < data.vocabulary.Length; i++)
            {
                sb.Append($"    \"{EscapeJsonString(data.vocabulary[i])}\"");
                if (i < data.vocabulary.Length - 1) sb.Append(",");
                sb.AppendLine();
            }
        }
        sb.AppendLine("  ],");
        sb.AppendLine($"  \"vocabSize\": {data.vocabSize},");
        sb.AppendLine($"  \"neuronCount\": {data.neuronCount},");
        sb.AppendLine($"  \"outputStart\": {data.outputStart},");
        sb.AppendLine($"  \"padToken\": {data.padToken},");
        sb.AppendLine($"  \"sosToken\": {data.sosToken},");
        sb.AppendLine($"  \"eosToken\": {data.eosToken},");
        sb.AppendLine($"  \"unkToken\": {data.unkToken},");
        sb.AppendLine($"  \"rnnDecay\": {data.rnnDecay.ToString("F6", System.Globalization.CultureInfo.InvariantCulture)},");
        sb.AppendLine($"  \"inputStart\": {data.inputStart},");
        sb.AppendLine($"  \"biasIndex\": {data.biasIndex},");

        sb.AppendLine("  \"order\": [");
        if (data.order != null)
        {
            for (int i = 0; i < data.order.Length; i++)
            {
                sb.Append($"    {data.order[i]}");
                if (i < data.order.Length - 1) sb.Append(",");
                sb.AppendLine();
            }
        }
        sb.AppendLine("  ],");

        sb.AppendLine("  \"neuronHasFFIncoming\": [");
        if (data.neuronHasFFIncoming != null)
        {
            for (int i = 0; i < data.neuronHasFFIncoming.Length; i++)
            {
                sb.Append($"    {(data.neuronHasFFIncoming[i] ? "true" : "false")}");
                if (i < data.neuronHasFFIncoming.Length - 1) sb.Append(",");
                sb.AppendLine();
            }
        }
        sb.AppendLine("  ],");

        sb.AppendLine("  \"connections\": [");
        if (data.connections != null)
        {
            for (int i = 0; i < data.connections.Length; i++)
            {
                var conn = data.connections[i];
                sb.Append($"    {{\"from\": {conn.from}, \"to\": {conn.to}, \"weight\": {conn.weight.ToString("F6", System.Globalization.CultureInfo.InvariantCulture)}, \"enabled\": {(conn.enabled ? "true" : "false")}, \"recurrent\": {(conn.recurrent ? "true" : "false")}}}");
                if (i < data.connections.Length - 1) sb.Append(",");
                sb.AppendLine();
            }
        }
        sb.AppendLine("  ]");
        sb.Append("}");
        return sb.ToString();
    }

    private string EscapeJsonString(string str)
    {
        if (str == null) return "";
        return str.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "\\r");
    }

    private float CalculateDistance(AI_NonUnity ai1, AI_NonUnity ai2)
    {
        List<int> innov1 = new List<int>();
        List<int> innov2 = new List<int>();
        Dictionary<int, float> weightMap1 = new Dictionary<int, float>();
        Dictionary<int, float> weightMap2 = new Dictionary<int, float>();

        for (int i = 0; i < ai1.innovations.Count; i++)
        {
            if (ai1.actConnect[i])
            {
                innov1.Add(ai1.innovations[i]);
                weightMap1[ai1.innovations[i]] = ai1.weights[i];
            }
        }
        for (int i = 0; i < ai2.innovations.Count; i++)
        {
            if (ai2.actConnect[i])
            {
                innov2.Add(ai2.innovations[i]);
                weightMap2[ai2.innovations[i]] = ai2.weights[i];
            }
        }

        innov1.Sort();
        innov2.Sort();

        int i1 = 0, i2 = 0;
        int disjoint = 0, excess = 0;
        float weightDiffSum = 0f;
        int matchingCount = 0;

        while (i1 < innov1.Count && i2 < innov2.Count)
        {
            if (innov1[i1] == innov2[i2])
            {
                matchingCount++;
                weightDiffSum += Math.Abs(weightMap1[innov1[i1]] - weightMap2[innov2[i2]]);
                i1++; i2++;
            }
            else if (innov1[i1] < innov2[i2])
            {
                disjoint++;
                i1++;
            }
            else
            {
                disjoint++;
                i2++;
            }
        }

        excess += (innov1.Count - i1) + (innov2.Count - i2);

        int N = Math.Max(innov1.Count, innov2.Count);
        if (N < 20) N = 1;

        float avgWeightDiff = (matchingCount > 0) ? weightDiffSum / matchingCount : 0f;
        float distance = (c1 * excess) / N + (c2 * disjoint) / N + c3 * avgWeightDiff;
        return distance;
    }

    /// <summary>
    /// Dynamically compute compatibility threshold based on population and genome complexity.
    /// Larger populations and more complex genomes need larger thresholds to maintain speciation.
    /// </summary>
    private float ComputeCompatibilityThreshold()
    {
        if (AIs.Count == 0) return baseCompatibilityThreshold;

        // Calculate average genome size (number of connections)
        float avgGenomeSize = (float)AIs.Average(a => a.innovations.Count);

        // Calculate population scaling factor
        // Aim for 2-5% of population in each species (20-50 species for population of 1000)
        int speciesTarget = Math.Max(3, AIs.Count / 50);

        // Scale threshold: base + adjustment for genome complexity
        // Larger genomes → larger threshold needed to maintain diversity
        float scaledThreshold = baseCompatibilityThreshold * (1.0f + (avgGenomeSize / 500.0f));

        logger?.Debug($"Dynamic threshold: base={baseCompatibilityThreshold:F2}, genomes={avgGenomeSize:F1}, scaled={scaledThreshold:F2}, target_species={speciesTarget}");

        return scaledThreshold;
    }

    public float GetBestFitness()
    {
        return AIs.Count > 0 ? AIs.Max(x => x.howIsGood) : 0f;
    }

    public float GetAverageFitness()
    {
        return AIs.Count > 0 ? AIs.Average(x => x.howIsGood) : 0f;
    }

    public int GetSpeciesCount()
    {
        return classAIs.Count;
    }

    public int GetLearnedPhrasesCount()
    {
        return phrasesComponent != null ? phrasesComponent.lengthOfknown : 0;
    }

    public AI_NonUnity GetBestAgent()
    {
        return AIs.Count > 0 ? AIs.OrderByDescending(x => x.howIsGood).First() : null;
    }
}
