using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;

namespace HW4
{
   public class ID3LearningEx : ID3Learning
    {
        public ID3LearningEx(DecisionTree tree) : base(tree) { }

        public new int MaxHeight
        {
            get { return base.MaxHeight; }
            set
            {
                // Ugly workaround, due to this BUG: https://github.com/accord-net/framework/issues/215
                // Fixed here, but not yet released: https://github.com/accord-net/framework/commit/839fe0d4383e10af7fbdbd4166656fed72bd9592 
                if (value <= 0)
                    throw new ArgumentOutOfRangeException("value", "The height must be greater than zero.");
                // ReSharper disable once PossibleNullReferenceException
                GetType().BaseType.GetField("maxHeight", BindingFlags.NonPublic | BindingFlags.Instance)?.SetValue(this, value);
            }
        }
    }

    public class Ensemble
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
            return yay > nay;
        }
    }
}
