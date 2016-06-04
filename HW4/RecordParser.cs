using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HW4
{
    public class RecordParser
    {
        public List<Record> DiscretizeDataset(List<Record> dataset, ReferenceTable table)
        {
            int[] row = { 0 };
            Discretize(table.HasUnknowns(0), rec => double.Parse(rec[row[0]]), (rec, val) => rec[row[0]] = val, table.GetRanges(row[0]), table.GetValues(row[0]), dataset); row[0]++;
            Discretize(table.HasUnknowns(0), rec => double.Parse(rec[row[0]]), (rec, val) => rec[row[0]] = val, table.GetRanges(row[0]), table.GetValues(row[0]), dataset); row[0]++;
            Discretize(table.HasUnknowns(0), rec => double.Parse(rec[row[0]]), (rec, val) => rec[row[0]] = val, table.GetRanges(row[0]), table.GetValues(row[0]), dataset); row[0]++;
            Discretize(table.HasUnknowns(0), rec => double.Parse(rec[row[0]]), (rec, val) => rec[row[0]] = val, table.GetRanges(row[0]), table.GetValues(row[0]), dataset); row[0]++;
            Discretize(table.HasUnknowns(0), rec => double.Parse(rec[row[0]]), (rec, val) => rec[row[0]] = val, table.GetRanges(row[0]), table.GetValues(row[0]), dataset); row[0]++;
            Discretize(table.HasUnknowns(0), rec => double.Parse(rec[row[0]]), (rec, val) => rec[row[0]] = val, table.GetRanges(row[0]), table.GetValues(row[0]), dataset); row[0]++;
            Discretize(table.HasUnknowns(0), rec => double.Parse(rec[row[0]]), (rec, val) => rec[row[0]] = val, table.GetRanges(row[0]), table.GetValues(row[0]), dataset); row[0]++;
            Discretize(table.HasUnknowns(0), rec => double.Parse(rec[row[0]]), (rec, val) => rec[row[0]] = val, table.GetRanges(row[0]), table.GetValues(row[0]), dataset); row[0]++;

            return dataset;
        }

        static void Discretize(bool hasUnknowns, Func<Record, double> getter, Action<Record, string> setter,
            double[] upperBounds, string[] values, List<Record> instances)
        {
            foreach (var instance in instances)
            {
                double value = getter(instance);
                if (hasUnknowns && Math.Abs(value) <= double.Epsilon)
                    setter(instance, "?");
                else
                {
                    for (int i = 0; i < upperBounds.Length; i++)
                        setter(instance, value < upperBounds[i] ? values[i] : values[i + 1]);
                }
            }
        }

       public List<Record> ParseRecords(string[] lines)
        {
            var records = new List<Record>();
            foreach (var line in lines)
            {
                string[] attributes = line.Split(new[] { "," }, StringSplitOptions.RemoveEmptyEntries);
                records.Add(new Record(attributes, int.Parse(attributes.Last()) == 1));
            }
            return records;
        }
    }
}
