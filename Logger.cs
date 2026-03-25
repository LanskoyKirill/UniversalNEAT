using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

public enum LogLevel
{
    Debug = 0,
    Info = 1,
    Warning = 2,
    Error = 3
}

public class Logger
{
    private string logsDirectory;
    private StreamWriter executionWriter;
    private StreamWriter fitnessWriter;
    private StreamWriter generatedTextWriter;
    private StreamWriter warningsWriter;
    private LogLevel minLevel;
    private object lockObject = new object();
    private bool headerWritten = false;

    public Logger(string logDir = "logs", LogLevel minLogLevel = LogLevel.Info)
    {
        logsDirectory = logDir;
        minLevel = minLogLevel;

        try
        {
            if (!Directory.Exists(logsDirectory))
                Directory.CreateDirectory(logsDirectory);

            string modelsDir = Path.Combine(logsDirectory, "models");
            if (!Directory.Exists(modelsDir))
                Directory.CreateDirectory(modelsDir);

            // Initialize writers
            executionWriter = new StreamWriter(
                Path.Combine(logsDirectory, "execution.log"),
                append: true,
                encoding: Encoding.UTF8) { AutoFlush = true };

            fitnessWriter = new StreamWriter(
                Path.Combine(logsDirectory, "fitness.log"),
                append: true,
                encoding: Encoding.UTF8) { AutoFlush = true };

            generatedTextWriter = new StreamWriter(
                Path.Combine(logsDirectory, "generated_text.log"),
                append: true,
                encoding: Encoding.UTF8) { AutoFlush = true };

            warningsWriter = new StreamWriter(
                Path.Combine(logsDirectory, "warnings.log"),
                append: true,
                encoding: Encoding.UTF8) { AutoFlush = true };
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] Failed to initialize logger: {ex.Message}");
        }
    }

    public void Debug(string message)
    {
        if (minLevel <= LogLevel.Debug)
            Log(LogLevel.Debug, message);
    }

    public void Info(string message)
    {
        if (minLevel <= LogLevel.Info)
            Log(LogLevel.Info, message);
    }

    public void Warning(string message)
    {
        if (minLevel <= LogLevel.Warning)
            Log(LogLevel.Warning, message);
    }

    public void Error(string message)
    {
        if (minLevel <= LogLevel.Error)
            Log(LogLevel.Error, message);
    }

    private void Log(LogLevel level, string message)
    {
        lock (lockObject)
        {
            string timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
            string levelStr = level.ToString().ToUpper();
            string formattedMessage = $"[{timestamp}] [{levelStr}] {message}";

            // Console output
            ConsoleColor originalColor = Console.ForegroundColor;
            Console.ForegroundColor = GetColorForLevel(level);
            Console.WriteLine(formattedMessage);
            Console.ForegroundColor = originalColor;

            // File output
            try
            {
                executionWriter?.WriteLine(formattedMessage);

                // Also log warnings and errors to warnings file
                if (level >= LogLevel.Warning)
                    warningsWriter?.WriteLine(formattedMessage);
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[ERROR] Failed to write log: {ex.Message}");
                Console.ForegroundColor = originalColor;
            }
        }
    }

    public void LogFitness(int epoch, float bestFitness, float avgFitness, int speciesCount)
    {
        lock (lockObject)
        {
            try
            {
                // Write CSV header if this is the first fitness entry
                if (!headerWritten)
                {
                    fitnessWriter?.WriteLine("Epoch,BestFitness,AvgFitness,SpeciesCount,Timestamp");
                    headerWritten = true;
                }

                string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
                string csvLine = $"{epoch},{bestFitness:F6},{avgFitness:F6},{speciesCount},{timestamp}";
                fitnessWriter?.WriteLine(csvLine);
            }
            catch (Exception ex)
            {
                Error($"Failed to log fitness data: {ex.Message}");
            }
        }
    }

    public void LogGeneratedText(int epoch, float fitness, string targetPhrase, string generatedText)
    {
        lock (lockObject)
        {
            try
            {
                string logEntry = $"[Epoch {epoch}] Fitness: {fitness:F4} | Target: \"{targetPhrase}\" | Generated: \"{generatedText}\"";
                generatedTextWriter?.WriteLine(logEntry);
            }
            catch (Exception ex)
            {
                Error($"Failed to log generated text: {ex.Message}");
            }
        }
    }

    public void LogEpochSummary(int epoch, float bestFitness, float avgFitness, int speciesCount, double elapsedSeconds)
    {
        string summary = $"Epoch {epoch}: Completed with best fitness={bestFitness:F4}, " +
                        $"avg fitness={avgFitness:F4}, {speciesCount} species in {elapsedSeconds:F2}s";
        Info(summary);
        LogFitness(epoch, bestFitness, avgFitness, speciesCount);
    }

    public string GetLogsDirectory()
    {
        return Path.GetFullPath(logsDirectory);
    }

    public void Flush()
    {
        lock (lockObject)
        {
            executionWriter?.Flush();
            fitnessWriter?.Flush();
            generatedTextWriter?.Flush();
            warningsWriter?.Flush();
        }
    }

    public void Dispose()
    {
        lock (lockObject)
        {
            executionWriter?.Dispose();
            fitnessWriter?.Dispose();
            generatedTextWriter?.Dispose();
            warningsWriter?.Dispose();
        }
    }

    private ConsoleColor GetColorForLevel(LogLevel level)
    {
        return level switch
        {
            LogLevel.Debug => ConsoleColor.Gray,
            LogLevel.Info => ConsoleColor.White,
            LogLevel.Warning => ConsoleColor.Yellow,
            LogLevel.Error => ConsoleColor.Red,
            _ => ConsoleColor.White
        };
    }
}
