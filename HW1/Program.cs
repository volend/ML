using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

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
            DateTime start = DateTime.Now;
            const string defaultName = "training_subsetD.arff";
            string fullFilePath = args.Length == 0 ? $@"..\..\Resources\{defaultName}" : args[0];

            List<Element> elements;
            List<Header> headers;
            ParseDataFile(File.ReadLines(fullFilePath), GetRegex(H_PATTERN), GetRegex(V_PATTERN), SplitArgs, out headers, out elements);
            Parallel.ForEach(headers, hdr => AssignProbabilities(hdr, elements));

            TimeSpan startTime = DateTime.Now - start;
            Console.WriteLine(startTime.TotalMilliseconds);

            start = DateTime.Now;
            var tree = new DecisionTree(headers, elements);

            startTime = DateTime.Now - start;
            Console.WriteLine(startTime.TotalMilliseconds);
        }

        static Regex GetRegex(string pattern)
        {
            return new Regex(pattern, RegexOptions.Singleline | RegexOptions.Compiled);
        }

        // This method is called by multiple threads to modify the same list of elements, which is not synchronized
        // however, the it is thread safe due to mutation slicing: each invocation will modify exactly one index of the values
        static void AssignProbabilities(Header header, List<Element> elements)
        {
            var counts = new Dictionary<string /*value*/, double /*count*/>();
            Array.ForEach(header.Values, str => counts.Add(str, 0));
            foreach (var element in elements)
            {
                double count;
                if (counts.TryGetValue(element[header], out count))
                    counts[element[header]] = count + 1;
            }

            var picker = new ProbabilityPicker(ConvertToProbabilities(counts));

            foreach (var element in elements)
            {
                if (!counts.ContainsKey(element[header]))
                    element[header] = picker.Pick();
            }
        }

        static List<Tuple<string, double>> ConvertToProbabilities(Dictionary<string, double> counts)
        {
            double totals = counts.Values.Sum();

            return counts.Select(counter => new Tuple<string, double>(counter.Key, counter.Value / totals * 100)).ToList();
        }

        static void ParseDataFile(IEnumerable<string> lines, Regex hRegex, Regex vRegex, string[] splitOptions,
            out List<Header> headers, out List<Element> elements)
        {
            headers = new List<Header>();
            elements = new List<Element>();
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
                    ParseElement(splitOptions, elements, line);
            }
        }

        static void ParseElement(string[] splitOptions, List<Element> elements, string line)
        {
            string[] values = line.Split(splitOptions, StringSplitOptions.RemoveEmptyEntries);
            if (!values.Any()) return;
            bool positive = bool.Parse(values[values.Length - 1]);
            elements.Add(new Element(values, positive));
        }

        static void ParseHeader(Regex hRegex, Regex vRegex, string[] splitOptions, List<Header> headers, string line)
        {
            if (!line.StartsWith(HEADER)) return;

            Match hdr = hRegex.Match(line.Remove(0, HEADER.Length));
            if (!hdr.Success) return;

            Match val = vRegex.Match(line);
            Contract.Assert(val.Success);

            string name = hdr.Value;
            string[] values = val.Value.Split(splitOptions, StringSplitOptions.RemoveEmptyEntries);
            headers.Add(new Header(headers.Count, name, values));
        }
    }

    public class TreeNode
    {
        readonly List<TreeNode> children;
        readonly Header attribute;
        public TreeNode(Header attribute)
        {
            this.attribute = attribute;
            children = new List<TreeNode>(attribute.Values.Length);
            foreach (var value in attribute.Values)
            {
                children.Add(null);
            }
        }
    }

    public class DecisionTree
    {
        readonly List<Element> samples;
        readonly List<Header> headers;

        public DecisionTree(List<Header> headers, List<Element> samples)
        {
            this.headers = headers;
            this.samples = samples;
        }

        public void BuildTree()
        {
            Header bestHeader = GetBestHeader(samples, headers);
        }

        Header GetBestHeader(List<Element> sampleSet, List<Header> headerSet)
        {
            Header bestHeader = null;
            double highestGain = 0.0d;

            Parallel.ForEach(headerSet, header =>
            {
                double gain = CalculateGain(sampleSet, header);
                lock (headerSet)
                {
                    if (gain > highestGain)
                    {
                        highestGain = gain;
                        bestHeader = header;
                    }
                }
            });

            return bestHeader;
        }

        double CalculateEntropy(List<Element> sampleSet)
        {
            int positives = CountPositiveExamples(sampleSet);
            return CalculateEntropy(positives, sampleSet.Count - positives);
        }

        double CalculateEntropy(int positives, int negatives)
        {
            double total = positives + negatives;

            double positiveProbability = positives / total;
            double negativeProbability = negatives / total;

            positiveProbability = -positiveProbability * Math.Log(positiveProbability, 2);
            negativeProbability = -negativeProbability * Math.Log(negativeProbability, 2);

            return positiveProbability + negativeProbability;
        }

        int CountPositiveExamples(IEnumerable<Element> sampleSet) => sampleSet.Count(positive => positive);

        double CalculateGain(List<Element> samplesSet, Header header)
        {
            int positives = CountPositiveExamples(samplesSet);
            double entropyBefore = CalculateEntropy(positives, samplesSet.Count - positives);

            double entropyAfter = 0.0d;
            foreach (var group in samplesSet.GroupBy(sample => sample[header]))
            {
                int groupPositives = CountPositiveExamples(group);
                int groupTotals = group.Count();
                entropyAfter += CalculateEntropy(groupPositives, group.Count()) * groupTotals / samplesSet.Count;
            }

            Debug.Assert(entropyAfter <= entropyBefore);
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

    public class Element
    {
        readonly string[] values;

        public Element(string[] values, bool positive)
        {
            Contract.Requires(values != null && values.Length > 0);
            this.values = values;
            IsPositive = positive;
        }

        public bool IsPositive { get; }

        public static implicit operator bool(Element e) => e.IsPositive;

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

        public string this[Header attr]
        {
            get { return this[attr.Index]; }
            set { this[attr.Index] = value; }
        }
    }

    public class Header
    {
        public int Index { get; }
        public string Name { get; }
        public string[] Values { get; }

        public Header(int index, string name, string[] values)
        {
            Contract.Requires(name != null);
            Contract.Requires(values != null);

            Index = index;
            Name = name;
            Values = values;
        }
    }
}
