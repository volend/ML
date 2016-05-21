using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HW3
{    
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
