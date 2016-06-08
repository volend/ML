using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HW4
{
    public class ReferenceTable
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
                case 8: return "Has diabetes?";
                default: throw new IndexOutOfRangeException();
            }
        }

        public string[] Columns
        {
            get
            {
                var result = new string[9];
                for (int i = 0; i < result.Length; i++)
                    result[i] = GetName(i);
                return result;
            }
        }

        public bool HasUnknowns(int index)
        {
            switch (index)
            {
                case 0: return false;
                case 1: return false;
                case 2: return false;
                case 3: return false;
                case 4: return false;
                case 5: return false;
                case 6: return false;
                case 7: return false;
                default: throw new IndexOutOfRangeException();
            }
        }
    }
}
