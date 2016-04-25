using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using System.Security.Policy;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Accord.Statistics.Kernels;
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
            const string testSet = "testingD.arff";
            string trainingSetPath = args.Length == 0 ? $@"..\..\Resources\{trainingSetFile}" : args[0];
            string testSetPath = args.Length == 0 ? $@"..\..\Resources\{testSet}" : args[0];

            List<Record> trainingSet;
            List<DiscreteAttribute> attributes;
            ParseDataFile(File.ReadLines(trainingSetPath), GetRegex(H_PATTERN), GetRegex(V_PATTERN), SplitArgs, out attributes, out trainingSet);
            attributes = attributes.Except(new[] { attributes.Last() }).ToList();

            AssignProbabilitiesByClass(trainingSet, attributes);

            List<Record> validationSet = PullValidationSet(trainingSet, percent: 0.10d);
            Parallel.ForEach(attributes, hdr => AssignProbabilities(hdr, validationSet));

            DecisionTree tree = new DecisionTree(attributes, trainingSet);
            tree.Build(90);
            double error = PercentError(validationSet, tree);
            Console.WriteLine($"Tree.Size({tree.Size()}) with Precision = {90} has errorRate={error}");

            //for (int i = 90; i > 50; i--)
            //{
            //    tree.Build(i);
            //    double error = PercentError(validationSet, tree);
            //    Console.WriteLine($"Tree.Size({tree.Size()}) with Precision = {i} has errorRate={error}");
            //}
        }

        static void AssignProbabilitiesByClass(List<Record> trainingSet, List<DiscreteAttribute> attributes)
        {
            var groups = trainingSet.GroupBy(elem => elem.IsPositive);
            foreach (var group in groups)
                Parallel.ForEach(attributes, hdr => AssignProbabilities(hdr, @group.ToList()));
            Parallel.ForEach(attributes, hdr => AssignProbabilities(hdr, trainingSet));
        }

        static double PercentError(List<Record> validationSet, DecisionTree tree)
        {
            double positive = validationSet.Count(record => record.IsPositive);
            double predictedPositive = validationSet.AsParallel().Count(tree.Test);

            return Math.Abs(predictedPositive - positive) / positive * 100;
        }

        static List<Record> PullValidationSet(List<Record> trainingSet, double percent)
        {
            var validationSet = new List<Record>();
            Random rand = new Random();
            for (int i = trainingSet.Count - 1; i >= 0; i--)
            {
                if (rand.NextDouble() <= percent)
                {
                    validationSet.Add(trainingSet[i]);
                    trainingSet.RemoveAt(i);
                }
            }

            return validationSet;
        }

        static Regex GetRegex(string pattern)
        {
            return new Regex(pattern, RegexOptions.Singleline | RegexOptions.Compiled);
        }

        // This method is called by multiple threads to modify the same list of elements, which is not synchronized
        // however, the it is thread safe due to mutation slicing: each invocation will modify exactly one index of the values
        static void AssignProbabilities(DiscreteAttribute discreteAttribute, List<Record> elements)
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
        readonly Dictionary<string/*value*/, TreeNode> children;
        readonly DiscreteAttribute attribute;
        Func<bool> decision;
        double percentAccuracy;
        string label;

        public TreeNode(string value, List<DiscreteAttribute> attributes, List<Record> records, double desiredAccuracy)
        {
            Value = value;
            children = new Dictionary<string, TreeNode>();

            if (DecideTrue(records)) return;
            if (DecideFalse(records)) return;
            DecideByMajority(records);
            if (percentAccuracy >= desiredAccuracy)
                return;
            attribute = DecisionTree.GetBestHeader(records, attributes);
            if (!CheckBestAttribute(records))
                CreateBranches(attributes, records, desiredAccuracy);

        }

        public void Print()
        {
            Console.WriteLine($"{label}");
            foreach (TreeNode childNode in children.Values)
                childNode.Print();
        }

        public string Label => label;
        public string Value { get; }

        public bool Test(Record record)
        {
            if (attribute == null)
                return decision();

            TreeNode nextNode;
            return children.TryGetValue(record[attribute], out nextNode) ? nextNode.Test(record) : decision();
        }

        void CreateBranches(List<DiscreteAttribute> attributes, List<Record> records, double desiredAccuracy)
        {
            if (percentAccuracy >= desiredAccuracy) return;

            label += $@"=> Attribute={attribute.Name}";
            var groups = records.GroupBy(record => record[attribute]);
            Parallel.ForEach(groups, group =>
            {
                lock (children)
                    children.Add(group.Key, new TreeNode(group.Key, attributes.Except(new[] { attribute }).ToList(),
                                                         group.ToList(), desiredAccuracy));
            });
        }

        bool CheckBestAttribute(List<Record> records)
        {
            if (attribute != null) return false;

            label = decision() ? bool.TrueString : bool.FalseString;
            return true;
        }

        void DecideByMajority(List<Record> records)
        {
            double positive = records.Count(record => record);
            percentAccuracy = 100 * positive / records.Count;
            decision = () => percentAccuracy > 50;
        }

        bool DecideFalse(List<Record> records)
        {
            if (records.TrueForAll(record => !record.IsPositive))
            {
                percentAccuracy = 100;
                decision = () => false;
                label = bool.FalseString;
                return true;
            }
            return false;
        }

        bool DecideTrue(List<Record> records)
        {
            if (records.TrueForAll(record => record.IsPositive))
            {
                percentAccuracy = 100;
                decision = () => true;
                label = bool.TrueString;
                return true;
            }
            return false;
        }

        public int Size() => 1 + children.Values.Sum(child => child.Size());
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

        public void Build(double desiredCertainty)
        {
            root = new TreeNode("root", attributes, records, desiredCertainty);
        }

        public bool Test(Record record) => root.Test(record);

        public void Print() => root.Print();

        public int Size() => root.Size();

        public static DiscreteAttribute GetBestHeader(List<Record> recordsSet, List<DiscreteAttribute> headersSet)
        {
            if (!headersSet.Any()) return null;

            var gains = new ConcurrentDictionary<double/*gain*/, DiscreteAttribute>();
            Parallel.ForEach(headersSet, header =>
            {
                double gain = CalculateGain(recordsSet, header);
                lock (gains)
                {
                    gains.TryAdd(gain, header);
                }
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

            double entropyAfter = 0.0d;
            foreach (var group in samplesSet.GroupBy(sample => sample[discreteAttribute]))
            {
                int groupTotals = group.Count();
                int groupPositives = CountPositiveExamples(group);
                entropyAfter += CalculateEntropy(groupPositives, groupTotals - groupPositives) * groupTotals / samplesSet.Count;
            }

            return entropyBefore - entropyAfter;
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
    }

    public class Record
    {
        readonly string[] values;

        public Record(string[] values, bool positive)
        {
            Contract.Requires(values != null && values.Length > 0);
            this.values = values;
            IsPositive = positive;
        }

        public bool IsPositive { get; }

        public static implicit operator bool(Record e) => e.IsPositive;

        public string this[int index]
        {
            get
            {
                Contract.Requires(index >= 0);
                Contract.Requires(index < values.Length - 1);
                return values[index];
            }
            set
            {
                Contract.Requires(index >= 0);
                Contract.Requires(index < values.Length - 1);
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
        public string[] Values { get; }

        public DiscreteAttribute(int index, string name, string[] values)
        {
            Contract.Requires(name != null);
            Contract.Requires(values != null);

            Index = index;
            Name = name;
            Values = values;
        }
    }
}
