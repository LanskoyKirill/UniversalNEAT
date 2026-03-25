using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

public class Program
{
    // Configuration
    // NOTE: These are defaults. Override with command-line flags:
    //   --population N    (default: 100)
    //   --selection N     (default: 20)
    //   --epochs N        (default: 50) - also accepts --epoch
    //   --phrases N       (default: 1)
    private static int POPULATION_SIZE = 100;
    private static int SELECTION_SIZE = 20;
    private static int EPOCHS = 50;
    private static int INITIAL_PHRASES = 1;
    private static Random random = new Random();

    static void Main(string[] args)
    {
        // Initialize logger
        Logger logger = new Logger("logs", LogLevel.Info);
        logger.Info("=".PadRight(80, '='));
        logger.Info("NEAT Neural Evolution Text Generation System - Console Version");
        logger.Info("=".PadRight(80, '='));

        try
        {
            // Parse configuration from command line or use defaults
            ParseArguments(args);

            logger.Info($"Configuration: Population={POPULATION_SIZE}, Selection={SELECTION_SIZE}, Epochs={EPOCHS}");
            logger.Info($"Logging to: {logger.GetLogsDirectory()}");

            // Sample story text - replace with actual story or read from file
            string storyText = LoadStoryText();
            if (string.IsNullOrEmpty(storyText))
            {
                logger.Error("No story text provided. Please provide story in LoadStoryText() or via file.");
                return;
            }

            logger.Info($"Story text loaded: {storyText.Length} characters");

            // Initialize Phrases component
            Phrases phrasesComponent = new Phrases(storyText);
            phrasesComponent.SetLogging(false);
            var allPhrases = phrasesComponent.GetAllPhrases();
            logger.Info($"Parsed {allPhrases.Count} phrases from story");

            // Get English sentences for tokenizer
            List<string> englishSentences = new List<string>();
            foreach (var phrasePair in allPhrases)
            {
                if (phrasePair.Count >= 2)
                    englishSentences.Add(phrasePair[1]);
            }

            // Create tokenizer
            Tokenizer tokenizer = new Tokenizer(englishSentences);
            int vocabSize = tokenizer.GetVocabSize();
            logger.Info($"Tokenizer created with vocab size: {vocabSize}");

            // Create game manager
            GameManager_NonUnity gameManager = new GameManager_NonUnity(
                logger,
                POPULATION_SIZE,
                SELECTION_SIZE,
                INITIAL_PHRASES,
                random
            );
            gameManager.SetPhrases(phrasesComponent);
            gameManager.SetTokenizer(tokenizer);

            logger.Info("Initializing population...");
            gameManager.Initialize();
            logger.Info($"Population initialized with {POPULATION_SIZE} agents");

            // Main evolutionary loop
            logger.Info("Starting evolution...");
            Stopwatch overallStopwatch = Stopwatch.StartNew();

            for (int epoch = 0; epoch < EPOCHS; epoch++)
            {
                Stopwatch epochStopwatch = Stopwatch.StartNew();

                // Process one epoch
                gameManager.ProcessEpoch(epoch);

                epochStopwatch.Stop();
                double epochTime = epochStopwatch.Elapsed.TotalSeconds;

                // Log epoch summary
                float bestFitness = gameManager.GetBestFitness();
                float avgFitness = gameManager.GetAverageFitness();
                int speciesCount = gameManager.GetSpeciesCount();
                int learnedPhrases = gameManager.GetLearnedPhrasesCount();

                logger.LogEpochSummary(epoch, bestFitness, avgFitness, speciesCount, epochTime);
                logger.Info($"  Learned phrases: {learnedPhrases}/{phrasesComponent.GetAllPhrases().Count}");


                // Log best agent's generated text
                AI_NonUnity bestAgent = gameManager.GetBestAgent();
                if (bestAgent != null)
                {
                    logger.LogGeneratedText(
                        epoch,
                        bestAgent.howIsGood,
                        bestAgent.targetPhrase,
                        bestAgent.generatedText
                    );
                }

                // Progress indicator
                int progress = (int)((epoch + 1) / (float)EPOCHS * 100);
                Console.Write($"\rProgress: {progress}% ({epoch + 1}/{EPOCHS})");
            }
            Console.WriteLine();

            overallStopwatch.Stop();
            double totalTime = overallStopwatch.Elapsed.TotalSeconds;

            // Export best model
            logger.Info("Exporting best model...");
            string modelPath = Path.Combine(logger.GetLogsDirectory(), "models", "best_model.json");
            gameManager.ExportBestModel(modelPath);
            logger.Info($"Model exported to: {modelPath}");

            // Print summary statistics
            PrintSummary(gameManager, logger, totalTime);

            logger.Info("=".PadRight(80, '='));
            logger.Info("Evolution completed successfully!");
            logger.Info("=".PadRight(80, '='));
        }
        catch (Exception ex)
        {
            logger.Error($"Fatal error: {ex.Message}");
            logger.Error($"Stack trace: {ex.StackTrace}");
            Console.WriteLine($"\nFatal error occurred. Check logs for details.");
            Environment.Exit(1);
        }
        finally
        {
            logger.Flush();
            logger.Dispose();
        }
    }

    private static void ParseArguments(string[] args)
    {
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLower())
            {
                case "--population":
                    if (i + 1 < args.Length && int.TryParse(args[i + 1], out int pop))
                    {
                        POPULATION_SIZE = pop;
                        i++;
                    }
                    break;
                case "--selection":
                    if (i + 1 < args.Length && int.TryParse(args[i + 1], out int sel))
                    {
                        SELECTION_SIZE = sel;
                        i++;
                    }
                    break;
                case "--epochs":
                case "--epoch":  // Accept both singular and plural
                    if (i + 1 < args.Length && int.TryParse(args[i + 1], out int epochs))
                    {
                        EPOCHS = epochs;
                        i++;
                    }
                    break;
                case "--phrases":
                case "--phrase":  // Accept both singular and plural
                    if (i + 1 < args.Length && int.TryParse(args[i + 1], out int phrases))
                    {
                        INITIAL_PHRASES = phrases;
                        i++;
                    }
                    break;
            }
        }
    }

    private static string LoadStoryText()
    {
        // Try to load from file first
        string storyFile = "story.txt";
        if (File.Exists(storyFile))
        {
            try
            {
                return File.ReadAllText(storyFile);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Could not load story from {storyFile}: {ex.Message}");
            }
        }

        // Return sample story text if no file found
        return @"This is a simple story. The quick brown fox jumps over the lazy dog.
                Machine learning is fascinating. Neural networks are powerful tools.
                Evolution of systems can produce surprising results. Text generation is challenging.
                Natural language processing requires careful consideration. Deep learning models need data.
                Reinforcement learning teaches agents through rewards. Supervised learning requires labeled examples.";
    }

    private static void PrintSummary(GameManager_NonUnity gameManager, Logger logger, double totalTime)
    {
        logger.Info("");
        logger.Info("=== Evolution Summary ===");
        logger.Info($"Total time: {totalTime:F2}s");
        logger.Info($"Best fitness achieved: {gameManager.GetBestFitness():F4}");
        logger.Info($"Average fitness (final epoch): {gameManager.GetAverageFitness():F4}");
        logger.Info($"Species count (final epoch): {gameManager.GetSpeciesCount()}");
        logger.Info($"Population size: {POPULATION_SIZE}");
        logger.Info($"Selection size: {SELECTION_SIZE}");
        logger.Info("");
        logger.Info("Logs saved to:");
        logger.Info($"  - execution.log: All events and debug information");
        logger.Info($"  - fitness.log: Evolution metrics in CSV format");
        logger.Info($"  - generated_text.log: Best generated text per epoch");
        logger.Info($"  - warnings.log: Errors and warnings");
        logger.Info($"  - models/best_model.json: Best trained NEAT model");
    }
}
