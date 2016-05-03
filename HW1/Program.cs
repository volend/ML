using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Accord.Statistics.Analysis;
using Accord.Statistics.Testing;

namespace HW1
{
    public class Program
    {
        const bool RANDOMIZE_DATASETS = true;
        const int TOTAL_TESTS = 3;
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
            var results = new Dictionary<double /*doc*/, List<Tuple<ConfusionMatrix, int /*treeSize*/, int /*treeDepth*/>>>();
            for (int i = 0; i < TOTAL_TESTS; i++)
            {
                Console.WriteLine($"Test #({i}/{TOTAL_TESTS}) with RANDOMIZATION={RANDOMIZE_DATASETS}");
                var testResults = PerformSingleTest(trainingData, testData, fracCertainties);
                for (int index = 0; index < testResults.Length; index++)
                {
                    List<Tuple<ConfusionMatrix, int /*treeSize*/, int /*treeDepth*/>> value;
                    if (!results.TryGetValue(fracCertainties[index], out value))
                        results[fracCertainties[index]] = value = new List<Tuple<ConfusionMatrix, int, int>>();
                    value.Add(testResults[index]);
                }
            }

            int idx = 0;
            foreach (var kvp in results)
            {
                Print($"[{fracCertainties[idx]}] Accuracy", kvp.Value, x => x.Item1.Accuracy);
                Print($"[{fracCertainties[idx]}] StandardError", kvp.Value, x => x.Item1.StandardError);
                Print($"[{fracCertainties[idx]}] dTree.Size", kvp.Value, x => x.Item2);
                Print($"[{fracCertainties[idx]}] dTree.Depth", kvp.Value, x => x.Item3);
                idx++;
            }
        }

        static void Print<T>(string label, List<T> values, Func<T, double> expression)
        {
            Console.WriteLine($"{label}: Avg={values.Average(expression)}, Min={values.Min(expression)}, Max={values.Max(expression)}.");
        }

        static Tuple<ConfusionMatrix, int/*TreeSize */, int/*tree depth*/>[] PerformSingleTest(string[] trainingData, string[] testData, double[] fracCertainties)
        {
            Console.WriteLine($"{GetTimeStamp()} Loading training data...");
            //Extract training set
            List<DiscreteAttribute> attributes;
            List<Record> trainingSet = ExtractDataSet(trainingData, out attributes, randomize: RANDOMIZE_DATASETS);

            //PrintStats(trainingSet, attributes);
            // Assign missing attributes by probabilities by class
            Console.WriteLine($"{GetTimeStamp()} Handle misisng attributes...");
            //DecisionTree.AssignProbabilitiesByClass(attributes, trainingSet, false);
            Parallel.ForEach(attributes, attribute => DecisionTree.AssignProbabilities(attribute, trainingSet));

            Console.WriteLine($"{GetTimeStamp()} Loading test data...");
            List<DiscreteAttribute> testAttributes;
            List<Record> testSet = ExtractDataSet(testData, out testAttributes, randomize: RANDOMIZE_DATASETS);
            // Assign missing attributes by probabilities (Note: we are not using probabilities by class, as the aim is to predict the class)
            Console.WriteLine($"{GetTimeStamp()} Handle misisng attributes...");
            Parallel.ForEach(testAttributes, attribute => DecisionTree.AssignProbabilities(attribute, testSet.Union(trainingSet).ToList()));
            //DecisionTree.AssignProbabilitiesByClass(testAttributes, testSet.Union(trainingSet).ToList(), false);

            var accuracyStats = new Tuple<ConfusionMatrix, int, int>[fracCertainties.Length];
            DecisionTree dTree = new DecisionTree(attributes, trainingSet);

            for (int index = 0; index < fracCertainties.Length; index++)
            {
                var fracCertainty = fracCertainties[index];
                Console.WriteLine($"{GetTimeStamp()} Building decision tree...");
                dTree.Build(fracCertainty);
                var confusionMatrix = CalculateAccuracy(testSet, dTree);
                var confusionMatrix1 = CalculateAccuracy(trainingSet, dTree);
                accuracyStats[index] = new Tuple<ConfusionMatrix, int, int>(confusionMatrix, dTree.Size(), dTree.Depth());
                // Perform sanity check by validating the training set
                Console.WriteLine(
                    $"{GetTimeStamp()} \nPositivePredictiveValue={confusionMatrix.PositivePredictiveValue}%\n" +
                    $"NegativePredictiveValue={confusionMatrix.NegativePredictiveValue}%\n" +
                    $"with DOC={fracCertainty}\n" +
                    $"ID3.size={dTree.Size()} " +
                    $"and ID3.depth={dTree.Depth()}\n");

                //dTree.Print();
                //var ruleSet = new List<DecisionRule>();
                //dTree.BuildRuleSets(ruleSet);
                //foreach (var rulesGroup in ruleSet.GroupBy(rule => rule.Positive))
                //{
                //    Console.WriteLine($"Printing rules with Value = {rulesGroup.Key}");
                //    foreach (var rule in rulesGroup)
                //        Console.WriteLine($"Rule: {rule.Label}");
                //}
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
            return trainingSet.AsParallel().OrderBy(x => rand.NextDouble()).ToList();//.ThenByDescending(x => rand.NextDouble()).ToList(); // Completely randomize the data set
        }

        static List<Record> ExtractDataSet(IEnumerable<string> textLines, out List<DiscreteAttribute> attributes, bool randomize)
        {
            List<Record> trainingSet;
            ParseDataFile(textLines, GetRegex(H_PATTERN), GetRegex(V_PATTERN), SplitArgs, out attributes,
                out trainingSet);
            attributes = attributes.Except(new[] { attributes.Last() }).ToList(); // Remove the boolean column
            return randomize ? RandomizeDataSet(trainingSet) : trainingSet;
        }

        static ConfusionMatrix CalculateAccuracy(List<Record> validationSet, DecisionTree tree)
        {
            int truePositives = validationSet.AsParallel().Count(record => tree.Test(record) && record.IsPositive);
            int falsePositives = validationSet.AsParallel().Count(record => tree.Test(record) && !record.IsPositive);
            int trueNegatives = validationSet.AsParallel().Count(record => !tree.Test(record) && !record.IsPositive);
            int falseNegatives = validationSet.AsParallel().Count(record => !tree.Test(record) && record.IsPositive);
            return new ConfusionMatrix(truePositives, falseNegatives, falsePositives, trueNegatives);
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
        readonly TreeNode parent;
        readonly ConcurrentDictionary<string, TreeNode> children;
        readonly DiscreteAttribute splitAttribute;
        Func<bool> decision;

        public TreeNode(TreeNode parent, string value, List<DiscreteAttribute> attributes, List<Record> records, double fracCertainty)
        {
            this.parent = parent;
            Value = value;
            children = new ConcurrentDictionary<string, TreeNode>();
            if (DecideTrue(records)) return;
            if (DecideFalse(records)) return;

            DecideWithProbability(records);

            splitAttribute = DecisionTree.GetBestAttribute(records, attributes);
            if (IsLeafNode()) return;

            BuildChildNodes(attributes, records, fracCertainty);
        }

        public void Print(int level)
        {
            for (int i = 0; i < level; i++)
                Console.Write("\t");
            Console.WriteLine($"[Level={level}] {Label}");
            foreach (TreeNode childNode in children.Values)
                childNode.Print(level + 1);
        }

        public string Label { get; set; }

        public string Value { get; }

        public bool Test(Record instance)
        {
            if (splitAttribute == null)
                return decision();

            TreeNode nextNode;
            return children.TryGetValue(instance[splitAttribute], out nextNode) ? nextNode.Test(instance) : decision();
        }

        void BuildChildNodes(List<DiscreteAttribute> attributes, List<Record> records, double fracCertainty)
        {
            Label += $@"Value={Value} => Attribute={splitAttribute.Name}";
            var groups = records.GroupBy(record => record[splitAttribute]).ToList();

            if (groups.Count == 1) return;

            ChiSquareTest chiSquare = CalculateChiSquare(records.Count(rec => rec), records.Count, groups);
            chiSquare.Size = 1 - fracCertainty;

            if (!chiSquare.Significant) return;

            Parallel.ForEach(groups, group =>
            //groups.ForEach(group =>
            {
                children.TryAdd(group.Key, new TreeNode(this, group.Key, attributes.Except(new[] { splitAttribute }).ToList(),
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

        bool IsLeafNode() => splitAttribute == null;

        void DecideWithProbability(List<Record> records)
        {
            //decision = () => false;
            double positive = records.Count(record => record);
            decision = () => positive * 2 >= records.Count;
            //var random = new Random();
            //double pPositive = positive / records.Count;
            //decision = () => random.NextDouble() <= pPositive;
            Label = $"{Value} => [p={positive / records.Count}] {(decision() ? bool.TrueString : bool.FalseString)}";
        }

        bool DecideFalse(List<Record> records)
        {
            if (records.TrueForAll(record => !record.IsPositive))
            {
                decision = () => false;
                Label = bool.FalseString;
                Label = $"{Value} => {bool.FalseString}";
                return true;
            }
            return false;
        }

        bool DecideTrue(List<Record> records)
        {
            if (records.TrueForAll(record => record.IsPositive))
            {
                decision = () => true;
                Label = $"{Value} => {bool.TrueString}";
                return true;
            }
            return false;
        }

        public int Size() => 1 + children.Values.Sum(child => child.Size());

        public int Depth()
        {
            return 1 + (children.Any() ? children.Values.Max(node => node.Depth()) : 0);
        }

        public void BuildRuleSets(List<DecisionRule> ruleSets)
        {
            if (IsLeafNode())
            {
                TreeNode node = this; // A leaf node always has a parent unless it's the root
                //var expression = new Stack<string>();
                var ruleExpression = new DecisionRule($" then => {decision()}", x => decision(), decision());

                //expression.Push($" then => {decision()}");
                while (node.parent != null)
                {
                    //expression.Push($" if [Attribute {node.parent.splitAttribute.Name} has Value {node.Value}] and ");
                    var localNode = node;
                    ruleExpression =
                        ruleExpression.Prepend(new DecisionRule(
                                $" if [Attribute {node.parent.splitAttribute.Name} has Value {node.Value}] and ",
                                x => x[localNode.parent.splitAttribute] == localNode.Value));
                    node = node.parent;
                    //sb.Append($"and Attribute={Attribute}")
                }

                lock (ruleSets)
                {
                    ruleSets.Add(ruleExpression);
                }
            }
            else
            {
                foreach (var childNode in children.Values)
                    childNode.BuildRuleSets(ruleSets);
            }
        }
    }

    public class DecisionRule
    {
        public bool Positive { get; protected set; }
        public string Label { get; protected set; }
        public Func<Record, bool> Func { get; protected set; }

        public DecisionRule(string label, Func<Record, bool> func, bool positive = false)
        {
            Label = label;
            Func = func;
            Positive = positive;
        }

        public DecisionRule Prepend(DecisionRule rule)
        {
            rule.Positive = Positive;
            rule.Label += Label;
            rule.Func = x => rule.Func(x) && Func(x);
            return rule;
        }
    }

    public class DecisionTree
    {
        readonly List<Record> records;
        readonly List<DiscreteAttribute> attributes;
        TreeNode root;
        double fracCertainty;

        public DecisionTree(List<DiscreteAttribute> attributes, List<Record> records)
        {
            this.attributes = attributes;
            this.records = records;
        }

        public void Build(double doc)
        {
            fracCertainty = doc;
            root = new TreeNode(null, "root", attributes, records, fracCertainty);
        }

        public void BuildRuleSets(List<DecisionRule> ruleSets)
        {
            root.BuildRuleSets(ruleSets);
        }

        public bool Test(Record instance) => root.Test(instance);

        public void Print() => root.Print(0);

        public int Size() => root.Size();

        public int Depth() => root.Depth();

        public static DiscreteAttribute GetBestAttribute(List<Record> recordsSet, List<DiscreteAttribute> headersSet)
        {
            if (!headersSet.Any()) return null;

            double bestGain = 0;
            DiscreteAttribute bestAttribute = null;
            Parallel.ForEach(headersSet.ToArray(), header =>
            {
                var gain = CalculateGain(recordsSet, header);
                //double ratio = CalculateRatio(recordsSet, header);
                if (gain.Item1)
                //if (true)
                {
                    lock (typeof(DecisionTree))
                    {
                        if (gain.Item2> bestGain)
                        {
                            bestGain = gain.Item2;
                            bestAttribute = header;
                        }
                    }
                }
                else
                {
                    lock (headersSet)
                    {
                        headersSet.Remove(header);
                    }
                }
            });

            return bestAttribute;
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

        static Tuple<bool/*significant*/, double> CalculateGain(List<Record> samplesSet, DiscreteAttribute attribute)
        {

            double entropyAfter = 0;
            foreach (var @group in samplesSet.GroupBy(sample => sample[attribute]))
            {
                //if (attribute.Values.Contains(@group.Key))
                {
                    int groupTotals = @group.Count();
                    int groupPositives = CountPositiveExamples(@group);
                    entropyAfter += CalculateEntropy(groupPositives, groupTotals - groupPositives) * groupTotals /
                                    samplesSet.Count;
                }
            }

            int positives = CountPositiveExamples(samplesSet);
            double entropyBefore = CalculateEntropy(positives, samplesSet.Count - positives);

            return new Tuple<bool, double>(entropyAfter < entropyBefore, entropyBefore - entropyAfter);
        }
        static double CalculateRatio(List<Record> recordsSet, DiscreteAttribute header)
        {
            double result = 0.0d;
            foreach (var grouping in recordsSet.GroupBy(record => record[header]))
                result = result + CalcPartEntropy(grouping.Count(), recordsSet.Count);
            return result;
        }

        public static void AssignProbabilitiesByClass(List<DiscreteAttribute> attributes, List<Record> trainingSet, bool pruneOutliers)
        {
            var outliers = new List<DiscreteAttribute>();
            var groups = trainingSet.GroupBy(elem => elem.IsPositive);
            foreach (var group in groups)
                Parallel.ForEach(attributes, hdr => AssignProbabilities(hdr, group.ToList(), outliers));
            //Parallel.ForEach(attributes, attribute => AssignProbabilitiesByClass(attribute, trainingSet));
            if (pruneOutliers)
            {
                foreach (var attr in outliers)
                    attributes.Remove(attr);
            }
        }

        public static void AssignProbabilitiesByClass(DiscreteAttribute attribute, List<Record> trainingSet)
        {
            var groups = trainingSet.GroupBy(elem => elem.IsPositive);
            foreach (var group in groups)
                AssignProbabilities(attribute, group.ToList(), new List<DiscreteAttribute>());
        }


        // This method is called by multiple threads to modify the same list of elements and is not synchronized.
        // However, it is thread safe due to mutation slicing: each invocation will modify exactly one index of the values
        public static void AssignProbabilities(DiscreteAttribute attribute, List<Record> elements, List<DiscreteAttribute> outliers = null)
        {
            var counts = new Dictionary<string /*value*/, double /*count*/>();
            Array.ForEach(attribute.Values, str => counts.Add(str, 0));
            foreach (var element in elements)
            {
                double count;
                if (counts.TryGetValue(element[attribute], out count))
                    counts[element[attribute]] = count + 1;
            }

            var picker = new ProbabilityPicker(ConvertToProbabilities(counts));
            if (!picker.IsSignificant() && outliers != null)
            {
                lock (outliers)
                    outliers.Add(attribute);
            }

            foreach (var element in elements)
            {
                if (!counts.ContainsKey(element[attribute]))
                    element[attribute] = picker.Pick();
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

        public bool IsSignificant() => probabilities.Sum(x => x.Item2) > 50;

        // Convert to probabilities based on percentages in order to use a random number generator
        // (i.e. if start with A=10%, B=20% and D=70%, after the conversion values will be: A=10%, B=30%, D=100%)
        public string Pick()
        {
            double randomChance = random.NextDouble() * 100;
            double cumulative = 0.0d;
            string result = probabilities.First().Item1;
            foreach (Tuple<string, double> tuple in probabilities)
            {
                cumulative += tuple.Item2;
                if (!(randomChance < cumulative)) continue;
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
