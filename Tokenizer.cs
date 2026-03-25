using System.Collections.Generic;
using System.Linq;
using System.Text;

public class Tokenizer
{
    private Dictionary<string, int> vocab = new Dictionary<string, int>();
    private Dictionary<int, string> reverse = new Dictionary<int, string>();
    
    public const int PAD_TOKEN = 0;
    public const int SOS_TOKEN = 1;
    public const int EOS_TOKEN = 2;
    public const int UNK_TOKEN = 3;
    
    private int currentIndex = 4;
    
    public Tokenizer(List<string> englishSentences)
    {
        vocab["<PAD>"] = PAD_TOKEN;
        vocab["<SOS>"] = SOS_TOKEN;
        vocab["<EOS>"] = EOS_TOKEN;
        vocab["<UNK>"] = UNK_TOKEN;
        
        BuildVocabulary(englishSentences, vocab, ref currentIndex);
        CreateReverseDictionary();
    }
    
    private void BuildVocabulary(List<string> sentences, Dictionary<string, int> vocab, ref int index)
    {
        foreach (var sentence in sentences)
        {
            var words = CleanAndSplit(sentence);
            foreach (var word in words)
            {
                if (!vocab.ContainsKey(word))
                {
                    vocab[word] = index++;
                }
            }
        }
    }
    
    private List<string> CleanAndSplit(string text)
    {
        var cleaned = new StringBuilder();
        foreach (char c in text.ToLower())
        {
            if (char.IsLetterOrDigit(c) || c == ' ' || c == '\'')
            {
                cleaned.Append(c);
            }
            else if (char.IsPunctuation(c))
            {
                cleaned.Append(' ');
            }
        }
        
        return cleaned.ToString()
            .Split(' ', System.StringSplitOptions.RemoveEmptyEntries)
            .ToList();
    }
    
    public List<int> Tokenize(string sentence)
    {
        var tokens = new List<int> { SOS_TOKEN };
        var words = CleanAndSplit(sentence);
        
        foreach (var word in words)
        {
            if (vocab.TryGetValue(word, out int token))
            {
                tokens.Add(token);
            }
            else
            {
                tokens.Add(UNK_TOKEN);
            }
        }
        tokens.Add(EOS_TOKEN);
        
        return tokens;
    }
    
    public string Detokenize(List<int> tokens)
    {
        var words = new List<string>();
        foreach (var token in tokens)
        {
            if (token == EOS_TOKEN) break;
            if (token == SOS_TOKEN || token == PAD_TOKEN) continue;
            
            if (reverse.TryGetValue(token, out string word))
            {
                words.Add(word);
            }
        }
        return string.Join(" ", words);
    }
    
    private void CreateReverseDictionary()
    {
        foreach (var kvp in vocab)
        {
            reverse[kvp.Value] = kvp.Key;
        }
    }

    public string[] ExportVocabulary()
    {
        int maxId = currentIndex; // следующее свободное id
        string[] vocabArray = new string[maxId];
        for (int i = 0; i < maxId; i++)
        {
            vocabArray[i] = reverse.ContainsKey(i) ? reverse[i] : "<UNK>";
        }
        return vocabArray;
    }
    
    public static Tokenizer FromVocabulary(string[] vocabArray)
    {
        var tok = new Tokenizer(new List<string>());
        tok.vocab.Clear();
        tok.reverse.Clear();
        for (int i = 0; i < vocabArray.Length; i++)
        {
            string word = vocabArray[i];
            tok.vocab[word] = i;
            tok.reverse[i] = word;
        }
        tok.currentIndex = vocabArray.Length;
        return tok;
    }
    public int GetVocabSize() => vocab.Count;
}