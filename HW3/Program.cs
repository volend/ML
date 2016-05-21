using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Analysis;

namespace HW3
{
    class Program
    {
        public static void Main(string[] args)
        {
            //string folder31 = args.Length != 2 ? @"..\..\Resources\3.1\" : args[0];
            //RunSpamFilter(Path.Combine(folder31, "train"), Path.Combine(folder31, "test"));

            string folder33 = args.Length != 2 ? @"..\..\Resources\3.3\" : args[1];
            RunDiabetesTest(Path.Combine(folder33, "train"), Path.Combine(folder33, "test"));
        }

        static void RunDiabetesTest(string trainingSet, string testSet)
        {
            List<Record> trainingRecords = ParseRecords(File.ReadAllLines(trainingSet));
        }

        static List<Record> ParseRecords(string[] lines)
        {
            var records = new List<Record>();
            foreach (var line in lines)
            {
                string[] attributes = line.Split(new[] { "," }, StringSplitOptions.RemoveEmptyEntries);
                records.Add(new Record(attributes, int.Parse(attributes.Last()) == 1));
            }
            return records;
        }

        static void RunSpamFilter(string trainingFile, string testFile)
        {
            Console.WriteLine($"[{GetTimeStamp()}] Loading training file: {trainingFile}.");
            var trainEmails = ParseEmails(File.ReadAllLines(trainingFile));

            Console.WriteLine($"[{GetTimeStamp()}] Loading test file: {testFile}.");
            var testEmails = ParseEmails(File.ReadAllLines(testFile));

            Console.WriteLine($"[{GetTimeStamp()}] Building indices...");
            foreach (SpamFilter.SmoothingStyle style in Enum.GetValues(typeof(SpamFilter.SmoothingStyle)))
            {
                SpamFilter filter = new SpamFilter(trainEmails, style);
                Console.WriteLine("--------------------------------------------------");
                Console.WriteLine($"[{GetTimeStamp()}] Calculating results using smoothing style = {style}");

                var trainMatrix = filter.RunPredictions(trainEmails);
                var testMatrix = filter.RunPredictions(testEmails);

                Console.WriteLine("\nSanity test with training data: ");
                PrintResults(trainMatrix);
                Console.WriteLine("\nActual results from test data: ");
                PrintResults(testMatrix);
            }
        }

        static void PrintResults(ConfusionMatrix matrix)
        {
            Console.WriteLine($"Accuracy={matrix.Accuracy}\n" +
                              $"Precision={matrix.Precision}\n" +
                              $"Recall={matrix.Recall}\n" +
                              $"NegativePredictionRate={matrix.NegativePredictiveValue}\n" +
                              $"PositivePredictionRate={matrix.PositivePredictiveValue}\n" +
                              $"{matrix}");
        }

        public static List<Email> ParseEmails(string[] lines)
        {
            List<Email> result = new List<Email>(lines.Length);
            foreach (var line in lines)
            {
                string[] features = line.Split(new[] { " " }, StringSplitOptions.RemoveEmptyEntries);
                var words = new Dictionary<string, int>();
                for (int i = 2; i < features.Length - 1; i += 2)
                {
                    words[features[i]] = int.Parse(features[i + 1]);
                }

                result.Add(new Email(features[0], features[1] == "spam", words));
            }
            return result;
        }

        static string GetTimeStamp()
        {
            return $"[{DateTime.Now.ToString("HH:mm:ss.fff")}]";
        }
    }
}
