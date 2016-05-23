using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Accord.Statistics.Analysis;

namespace HW3
{
    public class SpamFilter
    {
        readonly ProbabilityIndex index;

        public SpamFilter(List<Email> trainingData, SmoothingStyle style)
        {
            index = new ProbabilityIndex(trainingData, style);
        }

        public ConfusionMatrix RunPredictions(List<Email> emails)
        {
            var truePositives = emails.Count(email => email.IsSpam && IsSpam(email));
            var trueNegatives = emails.Count(email => !email.IsSpam && !IsSpam(email));
            var falsePositives = emails.Count(email => !email.IsSpam && IsSpam(email));
            var falseNegatives = emails.Count(email => email.IsSpam && !IsSpam(email));

            ConfusionMatrix result = new ConfusionMatrix(truePositives, falseNegatives, falsePositives, trueNegatives);
            return result;
        }

        public bool IsSpam(Email email) => index.IsSpam(email);

        public enum SmoothingStyle
        {
            LaplaceAddOne,
            JelinekMercer,
        }

        class ProbabilityIndex
        {
            static readonly double Alpha = 1;
            static readonly double Lambda = 0.2;
            static readonly double Padding = 300;

            readonly int totalSpamWords;
            readonly int totalHamWords;
            Dictionary<string, double> SpamIndex { get; }
            Dictionary<string, double> HamIndex { get; }
            SmoothingStyle Style { get; }

            double SpamClass { get; }
            double HamClass { get; }

            public ProbabilityIndex(List<Email> emails, SmoothingStyle style)
            {
                Style = style;
                SpamIndex = new Dictionary<string, double>();
                HamIndex = new Dictionary<string, double>();
                totalSpamWords = emails.Where(email => email.IsSpam).Sum(email => email.Words.Sum(word => word.Value));
                totalHamWords = emails.Where(email => !email.IsSpam).Sum(email => email.Words.Sum(word => word.Value));

                BuildIndex(SpamIndex, emails.FindAll(doc => doc.IsSpam));
                BuildIndex(HamIndex, emails.FindAll(doc => !doc.IsSpam));
                SpamClass = CalcClassProbability(emails, true);
                HamClass = CalcClassProbability(emails, false);
            }


            Dictionary<string, int> CountWords(List<Email> emails)
            {
                var result = new Dictionary<string, int>();
                foreach (var email in emails)
                {
                    foreach (var word in email.Words)
                    {
                        int count;
                        if (!result.TryGetValue(word.Key, out count))
                            result[word.Key] = 0;
                        result[word.Key] += word.Value;
                    }
                }
                return result;
            }

            public bool IsSpam(Email email)
            {
                double pIsSpam = email.Words.Aggregate(SpamClass,
                    (current, word) => current * WordProbability(word.Key, word.Value, true));
                double pIsHam = email.Words.Aggregate(HamClass,
                    (current, word) => current * WordProbability(word.Key, word.Value, false));
                return pIsSpam > pIsHam;
            }

            double WordProbability(string word, int count, bool isSpam)
            {
                var index = isSpam ? SpamIndex : HamIndex;
                double probability;
                switch (Style)
                {
                    case SmoothingStyle.LaplaceAddOne:
                        return index.TryGetValue(word, out probability)
                            ? Math.Pow(probability, count)
                            : Math.Pow(Padding / (isSpam ? totalSpamWords : totalHamWords), count);

                    case SmoothingStyle.JelinekMercer:
                        return index.TryGetValue(word, out probability)
                            ? Math.Pow(probability, count)
                            : Padding / (totalHamWords + totalSpamWords);

                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }

            // ReSharper disable once UnusedMember.Local
            public double MutualInformation(string word, bool isSpam)
            {
                double pWordCond = WordProbability(word, 1, isSpam);
                double pWordTotal = WordProbability(word, 1, true) + WordProbability(word, 1, false);
                double pClass = isSpam ? SpamClass : HamClass;
                double pJoint = pWordCond * pClass;
                double mutialInformation = pJoint * Math.Log(pJoint / (pWordTotal * pClass));

                return mutialInformation;
            }

            void BuildIndex(Dictionary<string, double> index, List<Email> classOfEmails)
            {
                int classWords = classOfEmails.Sum(x => x.Words.Sum(word => word.Value));
                var distinctWords = classOfEmails.SelectMany(email => email.Words.Keys).Distinct().ToArray();
                foreach (var word in distinctWords)
                {
                    if (Style == SmoothingStyle.LaplaceAddOne)
                    {
                        index[word] = Padding * (classOfEmails.Sum(doc => doc[word]) + Alpha) / (classWords + Alpha * distinctWords.Length);
                    }
                    else if (Style == SmoothingStyle.JelinekMercer)
                    {
                        int wordFrequency = classOfEmails.Sum(email => GetOrDefault(email, word));
                        index[word] = Padding * (1 - Lambda) * classOfEmails.Sum(doc => doc[word]) / classWords + Lambda * wordFrequency / classWords;
                    }
                }
            }

            int GetOrDefault(Email email, string word)
            {
                int count;
                return email.Words.TryGetValue(word, out count) ? count : 0;
            }

            double CalcClassProbability(List<Email> emails, bool isSpam)
            {
                return (double)emails.Count(doc => doc.IsSpam == isSpam) / emails.Count;
            }
        }
    }
    public class Email
    {
        public Email(string id, bool isSpam, Dictionary<string, int> words)
        {
            Id = id;
            IsSpam = isSpam;
            Words = words;
        }

        public int this[string word]
        {
            get
            {
                int num;
                if (Words.TryGetValue(word, out num))
                    return num;
                return 0;
            }
        }

        public Dictionary<string, int> Words { get; }

        public bool IsSpam { get; }

        public string Id { get; }
    }
}