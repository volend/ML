using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using Accord.MachineLearning.DecisionTrees;
using Accord.Math;
using Accord.Statistics.Analysis;
using Accord.Statistics.Filters;

namespace HW4
{
    class Program
    {
        static void Main(string[] args)
        {
            List<Record> trainingSet;
            List<Record> testSet;
            var table = BuildDataSets(out trainingSet, out testSet);

            for (int depth = 1; depth <= 3; depth++)
            {
                RunSingleTest(trainingSet, table, testSet, 1, depth);
                RunSingleTest(trainingSet, table, testSet, 20, depth);
                RunSingleTest(trainingSet, table, testSet, 500, depth);
            }
        }

        static void RunSingleTest(List<Record> trainingSet, ReferenceTable table, List<Record> testSet, int ensembleCount, int depth)
        {
            Console.WriteLine($"\n[{GetTimeStamp()}] Runing test with Ensemble = {ensembleCount} and MaxDepth = {depth}\n");
            Ensemble ensemble = new Ensemble();
            Random picker = new Random();
            for (int i = 0; i < ensembleCount; i++)
            {
                Learner learner = LoadDecisionTree(SampledData(trainingSet, picker), table, depth);
                ensemble.AddVoter(learner.Predict);
            }

            ConfusionMatrix testResults = RunTest(ensemble.Test, testSet);
            PrintResults(testResults);
        }

        static ReferenceTable BuildDataSets(out List<Record> trainingSet, out List<Record> testSet)
        {
            const string training = @"..\..\Resources\3.1\diabetes_train.txt";
            const string test = @"..\..\Resources\3.1\diabetes_test.txt";

            ReferenceTable table = new ReferenceTable();
            var parser = new RecordParser();

            trainingSet = parser.ParseRecords(File.ReadAllLines(training));
            testSet = parser.ParseRecords(File.ReadAllLines(test));

            parser.DiscretizeDataset(trainingSet, table);
            parser.DiscretizeDataset(testSet, table);

            return table;
        }

        static void PrintResults(ConfusionMatrix matrix)
        {
            var bias = ((double)matrix.FalseNegatives + matrix.FalsePositives) /
                       (matrix.TrueNegatives + matrix.TruePositives +
                        matrix.FalseNegatives + matrix.FalsePositives);

            Console.WriteLine($"Accuracy={matrix.Accuracy}\n" +
                              $"Precision={matrix.Precision}\n" +
                              $"Recall={matrix.Recall}\n" +
                              $"NegativePredictionRate={matrix.NegativePredictiveValue}\n" +
                              $"PositivePredictionRate={matrix.PositivePredictiveValue}\n" +
                              $"Variance={matrix.Variance}\n" +
                              $"Bias={bias}\n" +
                              $"{matrix}");
        }

        static ConfusionMatrix RunTest(Func<Record, bool> predictor, List<Record> instances)
        {
            var truePositive = instances.Count(record => record.IsPositive && predictor(record));
            var trueNegative = instances.Count(record => !record.IsPositive && !predictor(record));
            var falsePositive = instances.Count(record => !record.IsPositive && predictor(record));
            var falseNegative = instances.Count(record => record.IsPositive && !predictor(record));

            return new ConfusionMatrix(truePositive, falseNegative, falsePositive, trueNegative);
        }

        public static Learner LoadDecisionTree(List<Record> trainingSet, ReferenceTable table, int depth)
        {
            DataTable data = new DataTable("Diabetes dataset");

            data.Columns.AddRange(Array.ConvertAll(table.Columns, x => new DataColumn(x)));

            foreach (var record in trainingSet)
                data.Rows.Add(Array.ConvertAll(record.Values, x => x as object));

            Codification codebook = new Codification(data, table.Columns);
            DataTable symbols = codebook.Apply(data);
            int[][] inputs = symbols.ToArray<int>(ExcludeLast(table.Columns));
            int[] outputs = symbols.ToArray<int>(table.Columns.Last());

            var attributes = new DecisionVariable[table.Columns.Length - 1];
            for (int i = 0; i < attributes.Length; i++)
                attributes[i] = new DecisionVariable(table.Columns[i], table.GetValues(i).Length);

            int classCount = 2;

            DecisionTree tree = new DecisionTree(attributes, classCount);
            ID3LearningEx id3Learning = new ID3LearningEx(tree) { MaxHeight = depth };

            id3Learning.Run(inputs, outputs);

            return new Learner(tree, codebook, table.Columns.Last());
        }

        static List<Record> SampledData(List<Record> trainingRecords, Random picker)
        {
            List<Record> results = new List<Record>();
            while (results.Count < trainingRecords.Count)
            {
                int index = picker.Next(trainingRecords.Count, int.MaxValue) % trainingRecords.Count;
                results.Add(trainingRecords[index]);
            }
            return results;
        }

        public class Learner
        {
            DecisionTree Tree { get; }
            Codification Codebook { get; }

            String Label { get; }


            public Learner(DecisionTree tree, Codification codebook, string label)
            {
                Tree = tree;
                Codebook = codebook;
                Label = label;
            }

            public bool Predict(Record instance)
            {
                return Translate(instance, Tree, Codebook, Label);
            }
        }

        static string[] ExcludeLast(string[] columns)
        {
            var result = new string[columns.Length - 1];
            Array.Copy(columns, result, result.Length);
            return result;
        }

        static bool Translate(Record instance, DecisionTree tree, Codification codebook, string label)
        {
            int[] inputs = codebook.Translate(ExcludeLast(instance.Values));
            string answer = codebook.Translate(label, tree.Compute(inputs));
            return answer == "1";
        }


        static string GetTimeStamp()
        {
            return $"[{DateTime.Now.ToString("HH:mm:ss.fff")}]";
        }
    }
}
