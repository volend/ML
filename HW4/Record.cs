using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HW4
{
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

        public string[] Values => values;

        public string this[DiscreteAttribute attr]
        {
            get { return this[attr.Index]; }
            set { this[attr.Index] = value; }
        }
    }

}
