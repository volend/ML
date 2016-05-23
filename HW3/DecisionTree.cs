using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Accord.Statistics.Testing;

namespace HW3
{
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

        public DecisionTree Build()
        {
            root = new TreeNode(null, "root", attributes, records);
            return this;
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
            headersSet.ForEach(header =>
            {
                var gain = CalculateGain(recordsSet, header);
                if (gain > bestGain)
                {
                    bestGain = gain;
                    bestAttribute = header;
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

        static double CalculateRatio(List<Record> recordsSet, DiscreteAttribute header)
        {
            double result = 0.0d;
            foreach (var grouping in recordsSet.GroupBy(record => record[header]))
                result = result + CalcPartEntropy(grouping.Count(), recordsSet.Count);
            return result;
        }

        static double CalculateGain(List<Record> samplesSet, DiscreteAttribute attribute)
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

            return entropyBefore - entropyAfter;
        }

        public static void AssignProbabilitiesByClass(List<DiscreteAttribute> attributes, List<Record> trainingSet, bool pruneOutliers)
        {
            var outliers = new List<DiscreteAttribute>();
            var groups = trainingSet.GroupBy(elem => elem.IsPositive);
            foreach (var group in groups)
                Parallel.ForEach(attributes, hdr => AssignProbabilities(hdr, group.ToList(), outliers));
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
    public class TreeNode
    {
        readonly TreeNode parent;
        readonly ConcurrentDictionary<string, TreeNode> children;
        readonly DiscreteAttribute splitAttribute;
        Func<bool> decision;

        public TreeNode(TreeNode parent, string value, List<DiscreteAttribute> attributes, List<Record> records)
        {
            this.parent = parent;
            Value = value;
            children = new ConcurrentDictionary<string, TreeNode>();
            if (DecideTrue(records)) return;
            if (DecideFalse(records)) return;

            DecideWithProbability(records);

            splitAttribute = DecisionTree.GetBestAttribute(records, attributes);
            if (IsLeafNode())
            {
                return;
            }

            BuildChildNodes(attributes, records);
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

        void BuildChildNodes(List<DiscreteAttribute> attributes, List<Record> records)
        {
            Label += $@"Value={Value} => Attribute={splitAttribute.Name}";
            var groups = records.GroupBy(record => record[splitAttribute]).ToList();

            if (groups.Count == 1) return;

            groups.ForEach(group =>
            {
                children.TryAdd(group.Key, new TreeNode(this, group.Key, attributes.Except(new[] { splitAttribute }).ToList(),
                                                     group.ToList()));
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
            int positive = records.Count(record => record);
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
}
