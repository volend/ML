using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HW3
{
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
}
