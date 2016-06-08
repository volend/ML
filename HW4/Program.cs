using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using System.Net.Http.Headers;
using System.Xml;
using Accord.MachineLearning.Bayes;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math;
using Accord.Statistics.Analysis;
using Accord.Statistics.Filters;
using Accord.Statistics.Kernels;
using Microsoft.SqlServer.Server;

namespace HW4
{
    class Program
    {
        public Program(string name)
        {
            Console.WriteLine($"[{GetTimeStamp()}] Runnig {name}.");
        }

        static void Main(string[] args)
        {
            Program id3Ensemble = new Program("ID3 Diabetes Test");
            id3Ensemble.RunID3Test();

            Program kernelSVM = new Program("Kernel SVM Diabetes Test");
            kernelSVM.RunSVMTest();

            Program naiveBayes = new Program("Naive Bayes Diabetes Test");
            naiveBayes.RunNBTest();
        }

        void RunNBTest()
        {
            List<Record> trainingSet;
            List<Record> testSet;
            var table = BuildDataSets(out trainingSet, out testSet);

            int[][] inputs;
            int[] outputs;
            var codebook = BuildCodebook(trainingSet, table, out inputs, out outputs);

            int[] symbolCounts = new int[codebook.Columns.Count - 1];
            for (int i = 0; i < symbolCounts.Length; i++)
                symbolCounts[i] = codebook[i].Symbols;
            int classCount = codebook.Columns.Last().Symbols; // 2!

            NaiveBayes target = new NaiveBayes(classCount, symbolCounts);
            target.Estimate(inputs, outputs);
            NBLearner learner = new NBLearner(this, target, codebook);

            ConfusionMatrix testResults = RunTest(learner.Predict, testSet);
            PrintResults(testResults);
        }

        void RunSVMTest()
        {
            List<Record> trainingSet;
            List<Record> testSet;
            var table = BuildDataSets(out trainingSet, out testSet);

            int[][] inputs;
            int[] outputs;
            var codebook = BuildCodebook(trainingSet, table, out inputs, out outputs);


            Console.WriteLine($"\n[{GetTimeStamp()}] Running Test with Linear Kernel: \n");
            RunSingleTest(inputs, outputs, codebook, new Linear(), testSet);
            Console.WriteLine($"\n[{GetTimeStamp()}] Running Test with Polynomial Kernel: \n");
            RunSingleTest(inputs, outputs, codebook, new Polynomial(3), testSet);
            Console.WriteLine($"\n[{GetTimeStamp()}] Running Test with Gaussian Kernel: \n");
            RunSingleTest(inputs, outputs, codebook, new Gaussian(0.1), testSet);
            Console.WriteLine($"\n[{GetTimeStamp()}] Running Test with Sigmoid Kernel: \n");
            RunSingleTest(inputs, outputs, codebook, new Sigmoid(0.01, 0.01), testSet);
        }

        void RunSingleTest(int[][] inputs, int[] outputs, Codification codebook, KernelBase kernel, List<Record> testSet)
        {
            KernelSupportVectorMachine machine = new KernelSupportVectorMachine(kernel, inputs[0].Length);
            SequentialMinimalOptimization m = new SequentialMinimalOptimization(machine, ToDoubles(inputs), Normalize(outputs));
            m.Run();

            SVMLearner learner = new SVMLearner(this, codebook, machine);

            ConfusionMatrix testResults = RunTest(learner.Predict, testSet);
            PrintResults(testResults);
        }

        int[] Normalize(int[] outputs)
        {
            for (int i = 0; i < outputs.Length; i++)
                if (outputs[i] == 0)
                    outputs[i] = -1;
            return outputs;
        }

        double[][] ToDoubles(int[][] inputs)
        {
            double[][] result = new double[inputs.Length][];
            for (int i = 0; i < result.Length; i++)
                result[i] = ToDoubles(inputs[i]);
            return result;
        }

        double[] ToDoubles(int[] inputs)
        {
            double[] result = new double[inputs.Length];
            Array.Copy(inputs, result, result.Length);
            return result;
        }

        void RunID3Test()
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

        void RunSingleTest(List<Record> trainingSet, ReferenceTable table, List<Record> testSet, int ensembleCount, int depth)
        {
            Console.WriteLine($"\n[{GetTimeStamp()}] Runing test with Ensemble = {ensembleCount} and MaxDepth = {depth}\n");
            Ensemble ensemble = new Ensemble();
            Random picker = new Random();
            for (int i = 0; i < ensembleCount; i++)
            {
                ID3Learner learner = LoadDecisionTree(SampledData(trainingSet, picker), table, depth);
                ensemble.AddVoter(learner.Predict);
            }

            ConfusionMatrix testResults = RunTest(ensemble.Test, testSet);
            PrintResults(testResults);
        }

        ReferenceTable BuildDataSets(out List<Record> trainingSet, out List<Record> testSet)
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

        void PrintResults(ConfusionMatrix matrix)
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

        ConfusionMatrix RunTest(Func<Record, bool> predictor, List<Record> instances)
        {
            var truePositive = instances.Count(record => record.IsPositive && predictor(record));
            var trueNegative = instances.Count(record => !record.IsPositive && !predictor(record));
            var falsePositive = instances.Count(record => !record.IsPositive && predictor(record));
            var falseNegative = instances.Count(record => record.IsPositive && !predictor(record));

            return new ConfusionMatrix(truePositive, falseNegative, falsePositive, trueNegative);
        }

        ID3Learner LoadDecisionTree(List<Record> trainingSet, ReferenceTable table, int depth)
        {
            int[][] inputs;
            int[] outputs;
            var codebook = BuildCodebook(trainingSet, table, out inputs, out outputs);

            var attributes = new DecisionVariable[table.Columns.Length - 1];
            for (int i = 0; i < attributes.Length; i++)
                attributes[i] = new DecisionVariable(table.Columns[i], table.GetValues(i).Length);

            int classCount = 2;

            DecisionTree tree = new DecisionTree(attributes, classCount);
            ID3LearningEx id3Learning = new ID3LearningEx(tree) { MaxHeight = depth };

            id3Learning.Run(inputs, outputs);

            return new ID3Learner(this, tree, codebook, table.Columns.Last());
        }

        Codification BuildCodebook(List<Record> trainingSet, ReferenceTable table, out int[][] inputs, out int[] outputs)
        {
            DataTable data = new DataTable("Diabetes dataset");

            data.Columns.AddRange(Array.ConvertAll(table.Columns, x => new DataColumn(x)));

            trainingSet.ForEach(each => data.Rows.Add(each.Values));

            Codification codebook = new Codification(data);
            DataTable symbols = codebook.Apply(data);
            inputs = symbols.ToArray<int>(ExcludeLast(table.Columns));
            outputs = symbols.ToArray<int>(table.Columns.Last());
            return codebook;
        }

        List<Record> SampledData(List<Record> trainingRecords, Random picker)
        {
            List<Record> results = new List<Record>();
            while (results.Count < trainingRecords.Count)
            {
                int index = picker.Next(trainingRecords.Count, int.MaxValue) % trainingRecords.Count;
                results.Add(trainingRecords[index]);
            }
            return results;
        }

        public class NBLearner
        {
            Program Parent { get; }

            NaiveBayes Target { get; }

            Codification Codebook { get; }

            public NBLearner(Program parent, NaiveBayes target, Codification codebook)
            {
                Parent = parent;
                Target = target;
                Codebook = codebook;
            }

            public bool Predict(Record instance)
            {
                int[] inputs = Codebook.Translate(Parent.ExcludeLast(instance.Values));
                int output = Target.Compute(inputs);

                string result = Codebook.Translate(Codebook.Columns.Last().ColumnName, output);
                return result == "1";
            }
        }

        public class SVMLearner
        {
            Program Parent { get; }

            Codification Codebook { get; }
            KernelSupportVectorMachine Machine { get; }

            public SVMLearner(Program parent, Codification codebook, KernelSupportVectorMachine machine)
            {
                Parent = parent;
                Machine = machine;
                Codebook = codebook;
            }

            public bool Predict(Record instance)
            {
                int[] inputs = Codebook.Translate(Parent.ExcludeLast(instance.Values));
                return Math.Sign(Machine.Compute(Parent.ToDoubles(inputs))) > 0;
            }
        }

        public class ID3Learner
        {
            Program Parent { get; }
            DecisionTree Tree { get; }
            Codification Codebook { get; }

            String Label { get; }

            public ID3Learner(Program parent, DecisionTree tree, Codification codebook, string label)
            {
                Parent = parent;
                Tree = tree;
                Codebook = codebook;
                Label = label;
            }

            public bool Predict(Record instance)
            {
                return Parent.Translate(instance, Tree, Codebook, Label);
            }
        }

        string[] ExcludeLast(string[] columns)
        {
            var result = new string[columns.Length - 1];
            Array.Copy(columns, result, result.Length);
            return result;
        }

        bool Translate(Record instance, DecisionTree tree, Codification codebook, string label)
        {
            int[] inputs = codebook.Translate(ExcludeLast(instance.Values));
            string answer = codebook.Translate(label, tree.Compute(inputs));
            return answer == "1";
        }


        string GetTimeStamp()
        {
            return $"[{DateTime.Now.ToString("HH:mm:ss.fff")}]";
        }
    }
}
