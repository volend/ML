using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using System.IO;
using System.Text.RegularExpressions;
using Microsoft.Win32;

namespace HW1
{
    public class Program
    {
        const string RELATION = "@relation";
        const string HEADER = "@attribute";
        const string DATA = "@data";
        const string H_PATTERN = @"\'[^']+'|[\w]+";
        const string V_PATTERN = @"{.*?}";
        static readonly string[] SplitOptions = { "{", "}", "," };

        static void Main(string[] args)
        {
            const string defaultName = "training_subsetD.arff";
            string path = Directory.GetCurrentDirectory();
            //TODO: use Iterator
            string[] lines = File.ReadAllLines($@"..\..\Resources\{defaultName}");

            var headers = BuildHeaders(lines, new Regex(H_PATTERN), new Regex(V_PATTERN), SplitOptions);
        }

        static Dictionary<string, Header> BuildHeaders(string[] lines, Regex hRegex, Regex vRegex, string[] splitOptions)
        {
            var headers = new Dictionary<string, Header>();
            foreach (var line in lines)
            {
                if (line.Contains(DATA))
                    break;

                if (!line.StartsWith(HEADER)) continue;

                Match hdr = hRegex.Match(line.Remove(0, HEADER.Length));
                if (!hdr.Success) continue;

                Match val = vRegex.Match(line);
                Contract.Assert(val.Success);

                string name = hdr.Value;
                string[] values = val.Value.Split(splitOptions, StringSplitOptions.RemoveEmptyEntries);

                headers.Add(name, new Header(headers.Count, name, values));

                Console.WriteLine(hdr.Value);
                Console.WriteLine(string.Join(", ", values));
            }

            return headers;
        }
    }

    class Header
    {
        public int GetIndex { get; }
        public string GetName { get; }
        public string[] GetValues { get; }

        public Header(int index, string name, string[] values)
        {
            Contract.Requires(name != null);
            Contract.Requires(values != null);

            GetIndex = index;
            GetName = name;
            GetValues = values;
        }
    }
}
