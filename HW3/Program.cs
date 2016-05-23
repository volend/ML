using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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

        static void RunDiabetesTest(string trainingFile, string testFile)
        {
            Console.WriteLine($"[{GetTimeStamp()}] Diabetes test:");

            ReferenceTable table = new ReferenceTable();
            Console.WriteLine($"[{GetTimeStamp()}] Loading training file: {trainingFile}.");
            List<Record> trainingRecords = DiscretizeDataset(ParseRecords(File.ReadAllLines(trainingFile)), table);
            Console.WriteLine($"[{GetTimeStamp()}] Loading test file: {trainingFile}.");
            List<Record> testRecords = DiscretizeDataset(ParseRecords(File.ReadAllLines(testFile)), table);

            var attributes = table.GetIndex().Select(idx => new DiscreteAttribute(idx, table.GetName(idx), table.GetValues(idx))).ToList();

            //DecisionTree.AssignProbabilitiesByClass(attributes, trainingRecords, false);
            attributes.ForEach(attribute => DecisionTree.AssignProbabilities(attribute, trainingRecords));
            attributes.ForEach(attribute => DecisionTree.AssignProbabilities(attribute, testRecords.Union(trainingRecords).ToList()));

            Console.WriteLine($"[{GetTimeStamp()}] Building Ensemble of ID3 decision trees...");
            Random sampler = new Random();
            var ensemble = new Ensemble();
            for (int i = 0; i <= 1000; i++)
            {
                DecisionTree tree = new DecisionTree(attributes, SampledData(trainingRecords, sampler)).Build();
                ensemble.AddVoter(tree.Test);

                if (i != 0 && i != 20 && i != 100 && i != 500 && i != 1000 && i != 2000) continue;

                ConfusionMatrix trainingMatrix = RunPredictions(trainingRecords, rec => rec.IsPositive, ensemble.Test);
                ConfusionMatrix testMatrix = RunPredictions(testRecords, rec => rec.IsPositive, ensemble.Test);

                Console.WriteLine("----------------------------------------------------------------");
                Console.WriteLine($"[{GetTimeStamp()}][Ensemble: {i}] Printing sanity results: ");
                PrintResults(trainingMatrix);
                Console.WriteLine($"[{GetTimeStamp()}][Ensemble: {i}] Printing prediction results: ");
                PrintResults(testMatrix);
            }
        }

        static ConfusionMatrix RunPredictions<TRecord>(List<TRecord> instances, Func<TRecord, bool> identifier, Func<TRecord, bool> predictor)
        {
            var truePositives = instances.Count(instance => identifier(instance) && predictor(instance));
            var trueNegatives = instances.Count(instance => !identifier(instance) && !predictor(instance));
            var falsePositives = instances.Count(instance => !identifier(instance) && predictor(instance));
            var falseNegatives = instances.Count(instance => identifier(instance) && !predictor(instance));

            ConfusionMatrix result = new ConfusionMatrix(truePositives, falseNegatives, falsePositives, trueNegatives);
            return result;
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

        static List<Record> DiscretizeDataset(List<Record> dataset, ReferenceTable table)
        {
            int[] row = { 0 };
            Discretize(table.HasUnknowns(0), rec => double.Parse(rec[row[0]]), (rec, val) => rec[row[0]] = val, table.GetRanges(row[0]), table.GetValues(row[0]), dataset); row[0]++;
            Discretize(table.HasUnknowns(0), rec => double.Parse(rec[row[0]]), (rec, val) => rec[row[0]] = val, table.GetRanges(row[0]), table.GetValues(row[0]), dataset); row[0]++;
            Discretize(table.HasUnknowns(0), rec => double.Parse(rec[row[0]]), (rec, val) => rec[row[0]] = val, table.GetRanges(row[0]), table.GetValues(row[0]), dataset); row[0]++;
            Discretize(table.HasUnknowns(0), rec => double.Parse(rec[row[0]]), (rec, val) => rec[row[0]] = val, table.GetRanges(row[0]), table.GetValues(row[0]), dataset); row[0]++;
            Discretize(table.HasUnknowns(0), rec => double.Parse(rec[row[0]]), (rec, val) => rec[row[0]] = val, table.GetRanges(row[0]), table.GetValues(row[0]), dataset); row[0]++;
            Discretize(table.HasUnknowns(0), rec => double.Parse(rec[row[0]]), (rec, val) => rec[row[0]] = val, table.GetRanges(row[0]), table.GetValues(row[0]), dataset); row[0]++;
            Discretize(table.HasUnknowns(0), rec => double.Parse(rec[row[0]]), (rec, val) => rec[row[0]] = val, table.GetRanges(row[0]), table.GetValues(row[0]), dataset); row[0]++;
            Discretize(table.HasUnknowns(0), rec => double.Parse(rec[row[0]]), (rec, val) => rec[row[0]] = val, table.GetRanges(row[0]), table.GetValues(row[0]), dataset); row[0]++;

            return dataset;
        }

        static void Discretize(bool hasUnknowns, Func<Record, double> getter, Action<Record, string> setter,
            double[] upperBounds, string[] values, List<Record> instances)
        {
            foreach (var instance in instances)
            {
                double value = getter(instance);
                if (hasUnknowns && Math.Abs(value) <= double.Epsilon)
                    setter(instance, "?");
                else
                {
                    for (int i = 0; i < upperBounds.Length; i++)
                        setter(instance, value < upperBounds[i] ? values[i] : values[i + 1]);
                }
            }
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

                var trainMatrix = RunPredictions(trainEmails, email => email.IsSpam, filter.IsSpam);
                var testMatrix = RunPredictions(testEmails, email => email.IsSpam, filter.IsSpam);

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

    class ReferenceTable
    {
        public IEnumerable<int> GetIndex()
        {
            return Enumerable.Range(0, 7);
        }

        public double[] GetRanges(int index)
        {
            switch (index)
            {
                case 0: return new[] { 3d, 6, 9 };
                case 1: return new[] { 140d };
                case 2: return new[] { 60d, 80, 90 };
                case 3: return new[] { 4.5d, 36.5 };
                case 4: return new[] { 166d };
                case 5: return new[] { 18.5, 24.9 };
                case 6: return new[] { 0.2, 0.7 };
                case 7: return new[] { 30d, 40, 50 };
                default: throw new IndexOutOfRangeException();
            }
        }

        public string[] GetValues(int index)
        {
            switch (index)
            {
                case 0: return new[] { "x < 3", "3 <= x < 6", "6 <= x < 9", "x > 9" };
                case 1: return new[] { "Normal", "High" };
                case 2: return new[] { "Low", "Normal", "Pre-High", "High" };
                case 3: return new[] { "Low", "Normal", "High" };
                case 4: return new[] { "Low", "Normal" };
                case 5: return new[] { "Underweight", "Normal", "Overweight" };
                case 6: return new[] { "Low", "Normal", "High" };
                case 7: return new[] { "x < 30", "30 <= x < 40", "40 <= x < 50", "x > 50" };
                default: throw new IndexOutOfRangeException();
            }
        }

        public string GetName(int index)
        {
            switch (index)
            {
                case 0: return "Num times pregnant";
                case 1: return "Glucose test";
                case 2: return "Diastolic BP";
                case 3: return "Skin fold";
                case 4: return "2hr insulin";
                case 5: return "BMI";
                case 6: return "D Ped Func";
                case 7: return "Age";
                default: throw new IndexOutOfRangeException();
            }
        }

        public bool HasUnknowns(int index)
        {
            switch (index)
            {
                case 0: return false;
                case 1: return true;
                case 2: return true;
                case 3: return true;
                case 4: return true;
                case 5: return true;
                case 6: return true;
                case 7: return true;
                default: throw new IndexOutOfRangeException();
            }
        }
    }

    class Ensemble
    {
        readonly List<Func<Record, bool>> voters;
        public Ensemble()
        {
            voters = new List<Func<Record, bool>>();
        }

        public void AddVoter(Func<Record, bool> voter) => voters.Add(voter);

        public bool Test(Record instance)
        {
            int yay = voters.Count(voter => voter(instance));
            int nay = voters.Count(voter => !voter(instance));
            if (voters.Count > 1 && yay - nay == 1)
            {
                Console.WriteLine("Close call.");
            }
            return yay > nay;
        }
    }
}
