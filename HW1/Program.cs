using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
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

            //Extract training set
            List<DiscreteAttribute> attributes;
            List<Record> trainingSet = RandomizeDataSet(ExtractDataSet(trainingSetPath, out attributes));
            // Assign missing attributes by probabilities by class
            AssignProbabilitiesByClass(trainingSet, attributes);

            List<DiscreteAttribute> testAttributes;
            List<Record> testSet = RandomizeDataSet(ExtractDataSet(testSetPath, out testAttributes));
            // Assign missing attributes by probabilities (Note: we are not using probabilities by class, as the aim is to predict the class)
            Parallel.ForEach(testAttributes, hdr => AssignProbabilities(hdr, testSet));

            DecisionTree dTree = new DecisionTree(attributes, trainingSet);
            double[] fracCertainties = { 0.99d, 0.95d, 0.90d, 0.80d, 0 };
            foreach (var fracCertainty in fracCertainties)
            {
                dTree.Build(fracCertainty);
                double predictionAccuracy = CalculateAccuracy(testSet, dTree);
                double sanityCheckAccuracy = CalculateAccuracy(trainingSet, dTree); // Perform sanity check by validating the training set
                Console.WriteLine($"Prediction% / Sanity% = {predictionAccuracy}% / {sanityCheckAccuracy}% " +
                                  $"with DOC={fracCertainty} " +
                                  $"and ID3.size={dTree.Size()} " +
                                  $"and ID3.depth={dTree.Depth()}");
            }
        }

        static List<Record> RandomizeDataSet(List<Record> trainingSet)
        {
            Random rand = new Random(); // Uses Environment.Tickcount which is effectively a random seed
            trainingSet = trainingSet.OrderBy(x => rand.NextDouble()).ToList(); // Completely randomize the data set
            return trainingSet;
        }

        static List<Record> ExtractDataSet(string dataSetPath, out List<DiscreteAttribute> attributes)
        {
            List<Record> trainingSet;
            ParseDataFile(File.ReadLines(dataSetPath), GetRegex(H_PATTERN), GetRegex(V_PATTERN), SplitArgs, out attributes,
                out trainingSet);
            attributes = attributes.Except(new[] { attributes.Last() }).ToList(); // Remove the boolean column
            return trainingSet;
        }

        static void AssignProbabilitiesByClass(List<Record> trainingSet, List<DiscreteAttribute> attributes)
        {
            var groups = trainingSet.GroupBy(elem => elem.IsPositive);
            foreach (var group in groups)
                Parallel.ForEach(attributes, hdr => AssignProbabilities(hdr, group.ToList()));
            //Parallel.ForEach(attributes, hdr => AssignProbabilities(hdr, trainingSet));
        }

        static double CalculateAccuracy(List<Record> validationSet, DecisionTree tree)
        {
            double positive = validationSet.Count(record => record.IsPositive);
            double predictedPositive = validationSet.AsParallel().Count(tree.Test);

            return 100 - Math.Abs(predictedPositive - positive) / positive * 100;
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
                //element[discreteAttribute] = picker.PickMax();
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

        public TreeNode(string value, List<DiscreteAttribute> attributes, List<Record> records, double fracCertainty)
        {
            Value = value;
            children = new Dictionary<string, TreeNode>();
            if (DecideTrue(records)) return;
            if (DecideFalse(records)) return;

            DecideByMajority(records);

            attribute = DecisionTree.GetBestHeader(records, attributes);
            if (!IsLeafNode())
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

            ChiSquareTest chiSquare = CalculateChiSquare(records.Count(rec => rec), records.Count, groups);
            chiSquare.Size = 1 - fracCertainty;

            if (!chiSquare.Significant) return;

            Parallel.ForEach(groups, group =>
            {
                lock (children)
                    children.Add(group.Key, new TreeNode(group.Key, attributes.Except(new[] { attribute }).ToList(),
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
        public string[] Values { get; }

        public DiscreteAttribute(int index, string name, string[] values)
        {
            Index = index;
            Name = name;
            Values = values;
        }
    }
}
