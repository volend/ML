using System;
using System.Collections.Generic;
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
            Dictionary<string, Header> headers;
            ParseDataFile(File.ReadLines(fullFilePath), new Regex(H_PATTERN), new Regex(V_PATTERN), SplitArgs, out headers, out elements);

            Parallel.ForEach(headers, hdr => AssignProbabilities(hdr, elements));
            TimeSpan startTime = DateTime.Now - start;
            Console.WriteLine(startTime.TotalMilliseconds);
        }

        // This method is called by multiple threads to modify the same list of elements, which is not synchronized
        // however, the it is thread safe due to mutation slicing: each invocation will modify exactly one index of the values
        static void AssignProbabilities(KeyValuePair<string, Header> header, List<Element> elements)
        {
            var counts = new Dictionary<string /*value*/, double /*count*/>();
            Array.ForEach(header.Value.Values, val => counts.Add(val, 0));
            foreach (var element in elements)
            {
                double count;
                if (counts.TryGetValue(element[header.Value], out count))
                    counts[element[header.Value]] = count + 1;
            }

            var picker = new ProbabilityPicker(ConvertToProbabilities(counts));

            foreach (var element in elements)
            {
                if (!counts.ContainsKey(element[header.Value]))
                    element[header.Value.Index] = picker.Pick();
            }
        }

        static List<Tuple<string, double>> ConvertToProbabilities(Dictionary<string, double> counts)
        {
            double totals = counts.Values.Sum();

            return counts.Select(counter => new Tuple<string, double>(counter.Key, counter.Value / totals * 100)).ToList();
        }

        static void ParseDataFile(IEnumerable<string> lines, Regex hRegex, Regex vRegex, string[] splitOptions,
            out Dictionary<string, Header> headers, out List<Element> elements)
        {
            headers = new Dictionary<string, Header>();
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

        static void ParseHeader(Regex hRegex, Regex vRegex, string[] splitOptions, Dictionary<string, Header> headers, string line)
        {
            if (!line.StartsWith(HEADER)) return;

            Match hdr = hRegex.Match(line.Remove(0, HEADER.Length));
            if (!hdr.Success) return;

            Match val = vRegex.Match(line);
            Contract.Assert(val.Success);

            string name = hdr.Value;
            string[] values = val.Value.Split(splitOptions, StringSplitOptions.RemoveEmptyEntries);
            headers.Add(name, new Header(headers.Count, name, values));
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

        public string this[Header attr] => this[attr.Index];
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
