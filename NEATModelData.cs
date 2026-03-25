using System;
using System.Collections.Generic;

[Serializable]
public class NeatModelData
{
    // Токенизатор
    public string[] vocabulary;
    public int padToken = 0;
    public int sosToken = 1;
    public int eosToken = 2;
    public int unkToken = 3;

    // Архитектура
    public int vocabSize;
    public int neuronCount;
    public int biasIndex = 0;
    public int inputStart = 2;
    public int outputStart;

    // Гены
    public ConnectionGene[] connections;

    // Гиперпараметры
    public int[] order;                     // топологический порядок нейронов
    public bool[] neuronHasFFIncoming;      // флаги для Tanh
    public float rnnDecay = 0.95f;          // коэффициент затухания
}

[Serializable]
public struct ConnectionGene
{
    public int from;
    public int to;
    public float weight;
    public bool enabled;
    public bool recurrent;
}