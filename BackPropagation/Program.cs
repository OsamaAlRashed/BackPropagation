using CsvHelper;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using static System.Math;

namespace BackPropagation
{
    public static class Program
    {
        public static int type = 1; 
        public static int NetworkType = 1; 
        static void Main(string[] args)
        {

            var Dataset = new List<DatasetAttributeClass>();
            int NumberOfInput, NumberOfOutput, NumberOfHiddenLayers, Epochs;
            double LearnRate, Eps;

            var path = Path.Combine(Directory.GetCurrentDirectory(), "water.csv");

            using (var reader = new StreamReader(path, Encoding.Default))
            using (var csv = new CsvReader(reader, System.Globalization.CultureInfo.CreateSpecificCulture("en-us")))
            {
                Dataset = csv.GetRecords<DatasetAttributeClass>().ToList();
            }

            Print("Type Of Network: (1: Multiple Layer, 2: Elman)");
            NetworkType = Int32.Parse(Console.ReadLine());

            Print("Type Of Activation Function: (1: Sigmoid, 2: Tansig, 3: Purlin)");
            if(NetworkType != 2)
            {
                type = Int32.Parse(Console.ReadLine());
            }
            else
            {
                Print("In Elman We Will Use 'Tansig' For Hidden Layer And 'Purlin' For Output Layer");
            }

            Print("Number Of Inputs: ");
            NumberOfInput = Int32.Parse(Console.ReadLine());

            Print("Number Of Outputs: ");
            NumberOfOutput = Int32.Parse(Console.ReadLine());

            Print("Number Of Hidden Layers: ");
            NumberOfHiddenLayers = NetworkType == 1 ? Int32.Parse(Console.ReadLine()) : 1;
            if(NetworkType == 2) Print("1 (Elman Always Had One Hidden Layer)");

            Print("Number Of Epochs: ");
            Epochs = Int32.Parse(Console.ReadLine());

            Print("Learn Rate: ");
            LearnRate = Double.Parse(Console.ReadLine());

            Print("Eps: ");
            Eps = Double.Parse(Console.ReadLine());
            
            double[] Inputs = new double[NumberOfInput];
            double[] DesiredOutputs = new double[NumberOfOutput];
            int[] HiddenLayersLen = new int[NumberOfHiddenLayers];
            int MaxHiddenLayersLen = Max(NumberOfInput, NumberOfOutput);

            for (int i = 0; i < NumberOfHiddenLayers; i++)
            {
                Print($"Hidden Layer {i + 1}: ");
                HiddenLayersLen[i] = Int32.Parse(Console.ReadLine());
                MaxHiddenLayersLen = Max(MaxHiddenLayersLen, HiddenLayersLen[i]);
            }

            double[][][] Weights = new double[NumberOfHiddenLayers + 1][][];
            double[][] Thetas = new double[NumberOfHiddenLayers + 1][]; 
            
            for(int i = 0; i < NumberOfHiddenLayers + 1; i++)
            {
                if (i == 0)
                {
                    var temp = NumberOfInput;
                    if (NetworkType == 2) temp += HiddenLayersLen[0]; 
                    Weights[i] = new double[temp][];
                }
                else Weights[i] = new double[HiddenLayersLen[i - 1]][]; 
                for (int j = 0; j < Weights[i].Length; j++)
                {
                    if (i == NumberOfHiddenLayers) Weights[i][j] = new double[NumberOfOutput];
                    else Weights[i][j] = new double[HiddenLayersLen[i]];
                    if(j == 0) Thetas[i] = new double[Weights[i][j].Length];
                    for (int k = 0; k < Weights[i][j].Length; k++)
                    {
                        if (i == 0)
                        {
                            Weights[i][j][k] = (NumberOfInput + (NetworkType == 2? HiddenLayersLen[0] : 0)).RandInitialize();
                        }
                        else if(i == NumberOfHiddenLayers)
                        {
                            Weights[i][j][k] = NumberOfOutput.RandInitialize();
                        }
                        else
                        {
                            Weights[i][j][k] = HiddenLayersLen[i].RandInitialize();
                        }
                        if(j == 0) Thetas[i][k] = RandInitialize(20);
                    }
                }
            }

            Backpropagation(Epochs, LearnRate , Dataset, Inputs , DesiredOutputs , HiddenLayersLen , MaxHiddenLayersLen , NumberOfHiddenLayers ,Weights , Thetas, Eps);
            

        }

        public static double RandInitialize(this int F)
        {
            //F = 20;
            Random random = new Random();
            double start = -24;
            double end = 24;
            double len = end - start + 1;
            return ((random.NextDouble() * len) + start) / (10 * F);
        }
        public static double RandDouble()
        {
            Random random = new Random();
            return random.NextDouble();
        }
        public static double Sigmoid(this double xx)
        {
            var t = (1 / (1 + Pow(E, -xx)));
            return t;
        }
        public static double Tansig(double x)
        {
            return Math.Exp(-x) / Math.Pow((1 + Math.Exp(-x)), 2);
        }
        public static double Purlin(double x)
        {
            return x == 0 ? 0 : (x > 1 ? 1 : -1);
        }
        public static double ActivateFunction(this double xx, int? t = null)
        {
            t = (t == null) ? type : t; 
            switch (t)
            {
                case 1:
                    return Sigmoid(xx);
                case 2:
                    return Tansig(xx);
                case 3:
                    return Purlin(xx);
                default:
                    return 0;
            }
        }
        public static void Print(string text, ConsoleColor color = ConsoleColor.White)
        {
            Console.ForegroundColor = color;
            Console.WriteLine(text);
            Console.ResetColor();
        }

        public static void Backpropagation(int Epochs , double LearnRate, List<DatasetAttributeClass> Dataset , double[] Inputs , double[] DesiredOutputs
                                           , int[] HiddenLayersLen , int MaxHiddenLayersLen , int NumberOfHiddenLayers ,double[][][] Weights
                                           , double[][] Thetas , double Eps)
        {
            for (int epoch = 1; epoch <= Epochs; epoch++)
            {
                double[][] CurrentOutput = new double[NumberOfHiddenLayers][];
                double[] Outputs = new double[DesiredOutputs.Length];
                double[] Context = NetworkType == 2? new double[HiddenLayersLen[0]] : new double[0];
                int idx = 0;
                Dataset.Shuffle();
                foreach (var row in Dataset.Take(3000))
                {
                    Inputs[0] = row.in1;
                    Inputs[1] = row.in2;
                    Inputs[2] = row.in3;
                    Inputs[3] = row.in4;
                    Inputs[4] = row.in5;
                    Inputs[5] = row.in6;
                    Inputs[6] = row.in7;
                    Inputs[7] = row.in8;
                    Inputs[8] = row.in9;


                    DesiredOutputs[0] = row.out1;
                    //DesiredOutputs[1] = row.out2;
                    //DesiredOutputs[2] = row.out3;
                    //DesiredOutputs[3] = row.out4;
                    //DesiredOutputs[4] = row.out5;


                    for (int hidden = 0; hidden < NumberOfHiddenLayers; hidden++)
                    {
                        CurrentOutput[hidden] = new double[HiddenLayersLen[hidden]];
                        if (hidden == 0)
                        {
                            for (int node = 0; node < HiddenLayersLen[hidden]; node++)
                            {
                                for (int backNode = 0; backNode < Inputs.Length ; backNode++)
                                {
                                    CurrentOutput[hidden][node] += Inputs[backNode] * Weights[hidden][backNode][node];
                                }
                                if(NetworkType == 2)
                                {
                                    for (int backNode = 0; backNode < Context.Length; backNode++)
                                    {
                                        CurrentOutput[hidden][node] += Context[backNode] * Weights[hidden][Inputs.Length + backNode][node];
                                    }
                                }
                                CurrentOutput[hidden][node] -= Thetas[hidden][node];
                                CurrentOutput[hidden][node] = CurrentOutput[hidden][node].ActivateFunction(NetworkType == 2? 2 : type);
                            }
                            if (NetworkType == 2)
                            {
                                for(int node = 0; node < Context.Length; node++)
                                {
                                    Context[node] = CurrentOutput[hidden][node];
                                }
                            }
                        }
                        else
                        {
                            for (int node = 0; node < HiddenLayersLen[hidden]; node++)
                            {
                                for (int backNode = 0; backNode < HiddenLayersLen[hidden - 1]; backNode++)
                                {
                                    CurrentOutput[hidden][node] += CurrentOutput[hidden - 1][backNode] * Weights[hidden][backNode][node];
                                }
                                CurrentOutput[hidden][node] -= Thetas[hidden][node];
                                CurrentOutput[hidden][node] = CurrentOutput[hidden][node].ActivateFunction();
                            }
                        }
                    }
                    for (int output = 0; output < DesiredOutputs.Length ; output++)
                    {
                        for (int backNode = 0; backNode < HiddenLayersLen[NumberOfHiddenLayers - 1]; backNode++)
                        {
                            Outputs[output] += CurrentOutput[NumberOfHiddenLayers - 1][backNode] * Weights[NumberOfHiddenLayers][backNode][output];
                        }
                        Outputs[output] -= Thetas[NumberOfHiddenLayers][output];
                        Outputs[output] = Outputs[output].ActivateFunction(NetworkType == 2? 3 : type);
                    }
                    double[] OutputsError = new double[DesiredOutputs.Length];
                    double[] OutputsDelta = new double[DesiredOutputs.Length];
                    double[][] HiddensError = new double[NumberOfHiddenLayers][];
                    double[][] HiddensDelta = new double[NumberOfHiddenLayers][];

                    for (int output = 0; output < DesiredOutputs.Length; output++)
                    {
                        OutputsError[output] = DesiredOutputs[output] - Outputs[output];
                        OutputsDelta[output] = Outputs[output] * (1 - Outputs[output]) * OutputsError[output];
                    }
                    for (int layer = NumberOfHiddenLayers - 1; layer >= 0; layer--)
                    {
                        HiddensError[layer] = new double[HiddenLayersLen[layer]];
                        HiddensDelta[layer] = new double[HiddenLayersLen[layer]];
                        for (int node = 0; node < HiddensDelta[layer].Length; node++)
                        {
                            if (layer == NumberOfHiddenLayers - 1)
                            {
                                for (int nextLayer = 0; nextLayer < DesiredOutputs.Length; nextLayer++)
                                {
                                    HiddensError[layer][node] += OutputsDelta[nextLayer] * Weights[layer + 1][node][nextLayer];
                                }
                            }
                            else
                            {
                                for (int nextLayer = 0; nextLayer < HiddenLayersLen[layer + 1]; nextLayer++)
                                {
                                    HiddensError[layer][node] += HiddensDelta[layer + 1][nextLayer] * Weights[layer + 1][node][nextLayer];
                                }
                            }
                            HiddensDelta[layer][node] = CurrentOutput[layer][node] * (1 - CurrentOutput[layer][node]) * HiddensError[layer][node];
                        }
                    }

                    for (int layer = 0; layer < NumberOfHiddenLayers + 1; layer++)
                    {
                        if (layer == NumberOfHiddenLayers)
                        {
                            for (int node = 0; node < DesiredOutputs.Length; node++)
                            {
                                for (int lastNode = 0; lastNode < HiddenLayersLen[layer - 1]; lastNode++)
                                {
                                    Weights[layer][lastNode][node] = Weights[layer][lastNode][node] + LearnRate * CurrentOutput[layer - 1][lastNode] * OutputsDelta[node];
                                }
                                Thetas[layer][node] += LearnRate * -1 * OutputsDelta[node];
                            }

                        }
                        else if (layer == 0)
                        {
                            for (int node = 0; node < HiddenLayersLen[0]; node++)
                            {
                                for (int lastNode = 0; lastNode < Inputs.Length; lastNode++)
                                {
                                    Weights[layer][lastNode][node] = Weights[layer][lastNode][node] + LearnRate * Inputs[lastNode] * HiddensDelta[layer][node];
                                }
                                if(NetworkType == 2 && idx != 0)
                                {
                                    for (int lastNode = 0; lastNode < Context.Length; lastNode++)
                                    {
                                        Weights[layer][lastNode + Inputs.Length][node] = Weights[layer][lastNode + Inputs.Length][node] + LearnRate * Context[lastNode] * HiddensDelta[layer][node];
                                    }
                                }
                                Thetas[layer][node] += LearnRate * -1 * HiddensDelta[layer][node];
                            }
                        }
                        else
                        {
                            for (int node = 0; node < HiddenLayersLen[layer]; node++)
                            {
                                for (int lastNode = 0; lastNode < HiddenLayersLen[layer - 1]; lastNode++)
                                {
                                    Weights[layer][lastNode][node] = Weights[layer][lastNode][node] + LearnRate * CurrentOutput[layer - 1][lastNode] * HiddensDelta[layer][node];
                                }
                                Thetas[layer][node] += LearnRate * -1 * HiddensDelta[layer][node];
                            }
                        }
                    }

                    if (epoch == 1000)
                    {
                        for(int i = 0; i < DesiredOutputs.Length; i++)
                        {
                            Print("DesiredOutputs: " + DesiredOutputs[i]);
                            Print("Outputs: " + Outputs[i]);
                        }
                    }
                    idx++; 
                }
            }
            int ans = 0; 
            foreach (var row in Dataset.Skip(3000))
            {
                Inputs[0] = row.in1;
                Inputs[1] = row.in2;
                Inputs[2] = row.in3;
                Inputs[3] = row.in4;
                Inputs[4] = row.in5;
                Inputs[5] = row.in6;
                Inputs[6] = row.in7;
                Inputs[7] = row.in8;
                Inputs[8] = row.in9;


                DesiredOutputs[0] = row.out1;
                //DesiredOutputs[1] = row.out2;
                //DesiredOutputs[2] = row.out3;
                //DesiredOutputs[3] = row.out4;
                //DesiredOutputs[4] = row.out5;


                double[][] CurrentOutput = new double[NumberOfHiddenLayers][];
                double[] Outputs = new double[DesiredOutputs.Length];
                for (int hidden = 0; hidden < NumberOfHiddenLayers; hidden++)
                {
                    CurrentOutput[hidden] = new double[HiddenLayersLen[hidden]];
                    if (hidden == 0)
                    {
                        for (int node = 0; node < HiddenLayersLen[hidden]; node++)
                        {
                            for (int backNode = 0; backNode < Inputs.Length; backNode++)
                            {
                                CurrentOutput[hidden][node] += Inputs[backNode] * Weights[hidden][backNode][node];
                            }
                            CurrentOutput[hidden][node] -= Thetas[hidden][node];
                            CurrentOutput[hidden][node] = CurrentOutput[hidden][node].ActivateFunction(NetworkType == 2? 2 : type);
                        }
                    }
                    else
                    {
                        for (int node = 0; node < HiddenLayersLen[hidden]; node++)
                        {
                            for (int backNode = 0; backNode < HiddenLayersLen[hidden - 1]; backNode++)
                            {
                                CurrentOutput[hidden][node] += CurrentOutput[hidden - 1][backNode] * Weights[hidden][backNode][node];
                            }
                            CurrentOutput[hidden][node] -= Thetas[hidden][node];
                            CurrentOutput[hidden][node] = CurrentOutput[hidden][node].ActivateFunction();
                        }
                    }
                }
                bool flag = true;
                for (int output = 0; output < DesiredOutputs.Length; output++)
                {
                   
                    for (int backNode = 0; backNode < HiddenLayersLen[NumberOfHiddenLayers - 1]; backNode++)
                    {
                        Outputs[output] += CurrentOutput[NumberOfHiddenLayers - 1][backNode] * Weights[NumberOfHiddenLayers][backNode][output];
                    }
                    Outputs[output] -= Thetas[NumberOfHiddenLayers][output];
                    Outputs[output] = Outputs[output].ActivateFunction(NetworkType == 2? 3 : type);
                    if (Abs(Outputs[output] - DesiredOutputs[output]) > Eps) flag = false; 
                }
                if (flag) ans++;
            }
            Console.WriteLine(ans);
            Console.WriteLine("Acc: {0}", 1.0 * ans / (Dataset.Count - 3000));

        }

        private static Random rng = new Random();

        public static void Shuffle<T>(this IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

    }
}
