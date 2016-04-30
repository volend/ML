using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using System.Security.AccessControl;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Accord.Statistics.Testing;

namespace HW1
{
    public class Program
    {
        //const string RELATION = "@relation";
        const string HEADER = "@attribute";
        const string DATA = "@data";
        const string H_PATTERN = @"\'[^']+'|[\w]+";
        const string V_PATTERN = @"{.*?}";
        static readonly string[] SplitArgs = { "{", "}", "," };

        static void Main(string[] args)
        {
            const string trainingSetFile = "training_subsetD.arff";
            const string testSetFile = "testingD.arff";
            string folder = args.Length != 1 ? @"..\..\Resources\" : args[0];

            string trainingSetPath = Path.Combine(folder, trainingSetFile);
            string testSetPath = Path.Combine(folder, testSetFile);

            string[] trainingData = File.ReadAllLines(trainingSetPath);
            string[] testData = File.ReadAllLines(testSetPath);

            double[] fracCertainties = { 0.99d, 0.95d, 0.90d, 0 };
            var results = new Dictionary<double /*fracCertainty*/, List<Tuple<double /*accuracy*/, double /*sanity*/>>>();
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine($"Test #{i}");
                var testResults = PerformSingleTest(trainingData, testData, fracCertainties);
                for (int index = 0; index < testResults.Length; index++)
                {
                    List<Tuple<double /*accuracy*/, double /*sanity*/>> value;
                    if (!results.TryGetValue(fracCertainties[index], out value))
                    {
                        results[fracCertainties[index]] = value = new List<Tuple<double, double>>();
                    }
                    value.Add(testResults[index]);
                }
            }

            foreach (var kvp in results)
                Console.WriteLine($"DOC={kvp.Key}, AvgAccuracy={kvp.Value.Average(x => x.Item1)}, AvgSanity={kvp.Value.Average(x => x.Item2)}");
        }

        static Tuple<double/*accuracy*/, double/*sanity*/>[] PerformSingleTest(string[] trainingData, string[] testData, double[] fracCertainties)
        {
            Console.WriteLine($"{GetTimeStamp()} Loading training data...");
            //Extract training set
            List<DiscreteAttribute> attributes;
            List<Record> trainingSet = RandomizeDataSet(ExtractDataSet(trainingData, out attributes));

            //PrintStats(trainingSet, attributes);
            // Assign missing attributes by probabilities by class
            Console.WriteLine($"{GetTimeStamp()} Handle misisng attributes...");
            DecisionTree.AssignProbabilitiesByClass(attributes, trainingSet);

            Console.WriteLine($"{GetTimeStamp()} Loading test data...");
            List<DiscreteAttribute> testAttributes;
            List<Record> testSet = RandomizeDataSet(ExtractDataSet(testData, out testAttributes));
            // Assign missing attributes by probabilities (Note: we are not using probabilities by class, as the aim is to predict the class)
            Console.WriteLine($"{GetTimeStamp()} Handle misisng attributes...");
            Parallel.ForEach(testAttributes, attribute => DecisionTree.AssignProbabilities(attribute, testSet));

            var accuracyStats = new Tuple<double, double>[fracCertainties.Length];
            DecisionTree dTree = new DecisionTree(attributes, trainingSet);
            for (int index = 0; index < fracCertainties.Length; index++)
            {
                var fracCertainty = fracCertainties[index];
                Console.WriteLine($"{GetTimeStamp()} Building decision tree...");
                dTree.Build(fracCertainty);
                double predictionAccuracy = CalculateAccuracy(testSet, dTree);
                double sanityCheckAccuracy = CalculateAccuracy(trainingSet, dTree);
                accuracyStats[index] = new Tuple<double, double>(predictionAccuracy, sanityCheckAccuracy);
                // Perform sanity check by validating the training set
                Console.WriteLine(
                    $"{GetTimeStamp()} Prediction% / Sanity% = {predictionAccuracy}% / {sanityCheckAccuracy}% " +
                    $"with DOC={fracCertainty} " +
                    $"and ID3.size={dTree.Size()} " +
                    $"and ID3.depth={dTree.Depth()}");
            }

            return accuracyStats;
        }

        [SuppressMessage("ReSharper", "UnusedMember.Local")]
        static void PrintStats(List<Record> trainingSet, List<DiscreteAttribute> attributes)
        {
            int counter = 0;
            foreach (var attribute in attributes.OrderByDescending(attr => attr.Values.Length))
            {
                var groups = trainingSet.GroupBy(x => x[attribute]);
                Console.WriteLine($"[{counter++}]Attribute={attribute.Name}, Count={attribute.Values.Length}");
                foreach (var group in groups)
                    Console.Write($"[Count={group.Count()}] ");
                Console.WriteLine();
            }
        }

        static string GetTimeStamp()
        {
            return $"[{DateTime.Now.ToString("HH:mm:ss.fff")}]";
        }

        static List<Record> RandomizeDataSet(List<Record> trainingSet)
        {
            Random rand = new Random(); // Uses Environment.Tickcount which is effectively a random seed
            //trainingSet = trainingSet.OrderBy(x => rand.NextDouble()).ToList(); // Completely randomize the data set
            return trainingSet;
        }

        static List<Record> ExtractDataSet(IEnumerable<string> textLines, out List<DiscreteAttribute> attributes)
        {
            List<Record> trainingSet;
            ParseDataFile(textLines, GetRegex(H_PATTERN), GetRegex(V_PATTERN), SplitArgs, out attributes,
                out trainingSet);
            attributes = attributes.Except(new[] { attributes.Last() }).ToList(); // Remove the boolean column
            //foreach (var attribute in attributes)
            //    attribute.Values = new List<string>(attribute.Values) { "?" }.ToArray();
            return trainingSet;
        }

        static double CalculateAccuracy(List<Record> validationSet, DecisionTree tree)
        {
            double positive = validationSet.Count(record => record.IsPositive);
            double predictedPositive = validationSet.AsParallel().Count(tree.Test);

            return Math.Round(100 - Math.Abs(predictedPositive - positive) / positive * 100);
        }

        static Regex GetRegex(string pattern)
        {
            return new Regex(pattern, RegexOptions.Singleline | RegexOptions.Compiled);
        }

        static void ParseDataFile(IEnumerable<string> lines, Regex hRegex, Regex vRegex, string[] splitOptions,
            out List<DiscreteAttribute> headers, out List<Record> elements)
        {
            headers = new List<DiscreteAttribute>();
            elements = new List<Record>();
            bool headerSection = true;
            foreach (var line in lines)
            {
                if (headerSection && line.Contains(DATA))
                {
                    headerSection = false;
                    continue;
                }

                if (headerSection)
                    ParseHeader(hRegex, vRegex, splitOptions, headers, line);
                else
                    ParseRecord(splitOptions, elements, line);
            }
        }

        static void ParseRecord(string[] splitOptions, List<Record> elements, string line)
        {
            string[] values = line.Split(splitOptions, StringSplitOptions.RemoveEmptyEntries);
            if (!values.Any()) return;
            bool positive = bool.Parse(values[values.Length - 1]);
            elements.Add(new Record(values, positive));
        }

        static void ParseHeader(Regex hRegex, Regex vRegex, string[] splitOptions, List<DiscreteAttribute> headers, string line)
        {
            if (!line.StartsWith(HEADER)) return;

            Match hdr = hRegex.Match(line.Remove(0, HEADER.Length));
            if (!hdr.Success) return;

            Match val = vRegex.Match(line);
            Contract.Assert(val.Success);

            string name = hdr.Value;
            string[] values = val.Value.Split(splitOptions, StringSplitOptions.RemoveEmptyEntries);
            headers.Add(new DiscreteAttribute(headers.Count, name, values));
        }
    }

    public class TreeNode
    {
        readonly ConcurrentDictionary<string, TreeNode> children;
        readonly DiscreteAttribute attribute;
        Func<bool> decision;

        public TreeNode(string value, List<DiscreteAttribute> attributes, List<Record> records, double fracCertainty)
        {
            Value = value;
            children = new ConcurrentDictionary<string, TreeNode>();
            if (DecideTrue(records)) return;
            if (DecideFalse(records)) return;

            DecideByMajority(records);


            attribute = DecisionTree.GetBestAttribute(records, attributes);
            if (IsLeafNode()) return;

            //DecisionTree.AssignProbabilitiesByClass(attribute, records);
            BuildChildNodes(attributes, records, fracCertainty);
        }

        public void Print()
        {
            Console.WriteLine($"{Label}");
            foreach (TreeNode childNode in children.Values)
                childNode.Print();
        }

        public string Label { get; set; }

        public string Value { get; }

        public bool Test(Record instance)
        {
            if (attribute == null)
                return decision();

            TreeNode nextNode;
            return children.TryGetValue(instance[attribute], out nextNode) ? nextNode.Test(instance) : decision();
        }

        void BuildChildNodes(List<DiscreteAttribute> attributes, List<Record> records, double fracCertainty)
        {
            Label += $@"=> Attribute={attribute.Name}";
            var groups = records.GroupBy(record => record[attribute]).ToList();

            if (groups.Count == 1) return;

            ChiSquareTest chiSquare = CalculateChiSquare(records.Count(rec => rec), records.Count, groups);
            chiSquare.Size = 1 - fracCertainty;

            if (!chiSquare.Significant) return;

            Parallel.ForEach(groups, group =>
            {
                children.TryAdd(group.Key, new TreeNode(group.Key, attributes.Except(new[] { attribute }).ToList(),
                                                     group.ToList(), fracCertainty));
            });
        }

        ChiSquareTest CalculateChiSquare(int positive, int total, List<IGrouping<string, Record>> groups)
        {
            var pExpected = new double[groups.Count * 2];
            var pObserved = new double[groups.Count * 2];

            for (int i = 0; i < groups.Count; i++)
            {
                pExpected[i * 2] = (double)positive * groups[i].Count() / total;
                pExpected[i * 2 + 1] = (double)(total - positive) * groups[i].Count() / total;
                pObserved[i * 2] = groups[i].Count(record => record);
                pObserved[i * 2 + 1] = groups[i].Count(record => !record);
            }

            return new ChiSquareTest(pExpected, pObserved, groups.Count - 1);
        }

        bool IsLeafNode() => attribute == null;

        void DecideByMajority(List<Record> records)
        {
            double positive = records.Count(record => record);
            decision = () => 100 * positive / records.Count > 50;
            Label = decision() ? bool.TrueString : bool.FalseString;
        }

        bool DecideFalse(List<Record> records)
        {
            if (records.TrueForAll(record => !record.IsPositive))
            {
                decision = () => false;
                Label = bool.FalseString;
                return true;
            }
            return false;
        }

        bool DecideTrue(List<Record> records)
        {
            if (records.TrueForAll(record => record.IsPositive))
            {
                decision = () => true;
                Label = bool.TrueString;
                return true;
            }
            return false;
        }

        public int Size() => 1 + children.Values.Sum(child => child.Size());

        public int Depth()
        {
            return 1 + (children.Any() ? children.Values.Max(node => node.Depth()) : 0);
        }
    }

    public class DecisionTree
    {
        readonly List<Record> records;
        readonly List<DiscreteAttribute> attributes;
        TreeNode root;

        public DecisionTree(List<DiscreteAttribute> attributes, List<Record> records)
        {
            this.attributes = attributes;
            this.records = records;
        }

        public void Build(double fracCertainty)
        {
            root = new TreeNode("root", attributes, records, fracCertainty);
        }

        public bool Test(Record instance) => root.Test(instance);

        public void Print() => root.Print();

        public int Size() => root.Size();

        public int Depth() => root.Depth();

        public static DiscreteAttribute GetBestAttribute(List<Record> recordsSet, List<DiscreteAttribute> headersSet)
        {
            if (!headersSet.Any()) return null;

            var gains = new ConcurrentDictionary<double/*gain*/, DiscreteAttribute>();
            Parallel.ForEach(headersSet, header =>
            //headersSet.ForEach(header =>
            {
                double gain = CalculateGain(recordsSet, header);
                //double ratio = CalculateRatio(recordsSet, header);
                gains.TryAdd(gain, header);
            });

            return gains[gains.Keys.Max()];
        }

        public static double CalculateEntropy(List<Record> sampleSet)
        {
            int positives = CountPositiveExamples(sampleSet);
            return CalculateEntropy(positives, sampleSet.Count - positives);
        }

        static double CalculateEntropy(int positives, int negatives)
        {
            double total = positives + negatives;
            return CalcPartEntropy(positives, total) + CalcPartEntropy(negatives, total);
        }

        static double CalcPartEntropy(int part, double total)
        {
            if (part == 0) return 0; // According to .NET CLR: -Infinity * 0 = NaN
            double partProbability = part / total;
            return -partProbability * Math.Log(partProbability, 2);
        }

        static int CountPositiveExamples(IEnumerable<Record> sampleSet) => sampleSet.Count(positive => positive);

        static double CalculateGain(List<Record> samplesSet, DiscreteAttribute discreteAttribute)
        {
            int positives = CountPositiveExamples(samplesSet);
            double entropyBefore = CalculateEntropy(positives, samplesSet.Count - positives);

            double entropyAfter = (from @group in samplesSet.GroupBy(sample => sample[discreteAttribute])
                                   where discreteAttribute.Values.Contains(@group.Key)
                                   let groupTotals = @group.Count()
                                   let groupPositives = CountPositiveExamples(@group)
                                   select CalculateEntropy(groupPositives, groupTotals - groupPositives) * groupTotals / samplesSet.Count).Sum();

            return entropyBefore - entropyAfter;
        }
        static double CalculateRatio(List<Record> recordsSet, DiscreteAttribute header)
        {
            double result = 0.0d;
            foreach (var grouping in recordsSet.GroupBy(record => record[header]))
                result = result + CalcPartEntropy(grouping.Count(), recordsSet.Count);
            return result;
        }

        public static void AssignProbabilitiesByClass(List<DiscreteAttribute> attributes, List<Record> trainingSet)
        {
            //var groups = trainingSet.GroupBy(elem => elem.IsPositive);
            //foreach (var group in groups)
            Parallel.ForEach(attributes, attribute => AssignProbabilitiesByClass(attribute, trainingSet));
            //Parallel.ForEach(attributes, hdr => AssignProbabilities(hdr, trainingSet));
        }

        public static void AssignProbabilitiesByClass(DiscreteAttribute attribute, List<Record> trainingSet)
        {
            var groups = trainingSet.GroupBy(elem => elem.IsPositive);
            foreach (var group in groups)
                AssignProbabilities(attribute, group.ToList());
        }


        // This method is called by multiple threads to modify the same list of elements and is not synchronized.
        // However, it is thread safe due to mutation slicing: each invocation will modify exactly one index of the values
        public static void AssignProbabilities(DiscreteAttribute discreteAttribute, List<Record> elements)
        {
            var counts = new Dictionary<string /*value*/, double /*count*/>();
            Array.ForEach(discreteAttribute.Values, str => counts.Add(str, 0));
            foreach (var element in elements)
            {
                double count;
                if (counts.TryGetValue(element[discreteAttribute], out count))
                    counts[element[discreteAttribute]] = count + 1;
            }

            var picker = new ProbabilityPicker(ConvertToProbabilities(counts));

            foreach (var element in elements)
            {
                if (!counts.ContainsKey(element[discreteAttribute]))
                    element[discreteAttribute] = picker.Pick();
            }
        }

        static List<Tuple<string, double>> ConvertToProbabilities(Dictionary<string, double> counts)
        {
            double totals = counts.Values.Sum();

            return counts.Select(counter => new Tuple<string, double>(counter.Key, counter.Value / totals * 100)).ToList();
        }
    }

    public class ProbabilityPicker
    {
        readonly List<Tuple<string, double>> probabilities;
        readonly Random random;

        public ProbabilityPicker(List<Tuple<string, double>> probabilities)
        {
            this.probabilities = probabilities;
            random = new Random(1000);
        }

        // Convert to probabilities based on percentages in order to use a random number generator
        // (i.e. if start with A=10%, B=20% and D=70%, after the conversion values will be: A=10%, B=30%, D=100%)
        public string Pick()
        {
            double diceRoll = random.NextDouble() * 100;
            double cumulative = 0.0d;
            string result = probabilities.First().Item1;
            foreach (Tuple<string, double> tuple in probabilities)
            {
                cumulative += tuple.Item2;
                if (!(diceRoll < cumulative)) continue;
                result = tuple.Item1;
                break;
            }
            return result;
        }

        public string PickMax() => probabilities.Aggregate((l, r) => l.Item2 > r.Item2 ? l : r).Item1;
    }

    public class Record
    {
        readonly string[] values;

        public Record(string[] values, bool positive)
        {
            this.values = values;
            IsPositive = positive;
        }

        public bool IsPositive { get; }

        public static implicit operator bool(Record e) => e.IsPositive;

        public string this[int index]
        {
            get
            {
                return values[index];
            }
            set
            {
                values[index] = value;
            }
        }

        public string this[DiscreteAttribute attr]
        {
            get { return this[attr.Index]; }
            set { this[attr.Index] = value; }
        }
    }

    public class DiscreteAttribute
    {
        public int Index { get; }
        public string Name { get; }
        public string[] Values { get; set; }

        public DiscreteAttribute(int index, string name, string[] values)
        {
            Index = index;
            Name = name;
            Values = values;
        }
    }
}
