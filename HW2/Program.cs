using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace HW2
{
    class Program
    {
        static void Main(string[] args)
        {
            string folder = args.Length != 1 ? @"..\..\Resources\" : args[0];

            Console.WriteLine($"{GetTimeStamp()} Files must be in folder {folder}. Use the first argument to specify another folder.");
            Console.Write($"{GetTimeStamp()} Enter # of neighbors [50-10000]: ");
            int neighbors;
            if (!int.TryParse(Console.ReadLine(), out neighbors))
                Console.Write("Unable to parse input.");
            else neighbors = neighbors < 50 ? 50 : (neighbors > 10000 ? 10000 : neighbors);
            Console.WriteLine($"{GetTimeStamp()} neighbors set to: {neighbors}");

            Console.WriteLine($"{GetTimeStamp()} Loading data...");
            const string moviesFile = "movie_titles.txt";
            const string trainingRatingsFile = "TrainingRatings.txt";
            const string testRatingsFile = "TestingRatings.txt";

            // ReSharper disable once UnusedVariable
            MovieData movies = ParseMoviesData(File.ReadAllLines(Path.Combine(folder, moviesFile)));

            RatingData trainingRatings = ParseRatingData(File.ReadAllLines(Path.Combine(folder, trainingRatingsFile)));
            RatingData testRatings = ParseRatingData(File.ReadAllLines(Path.Combine(folder, testRatingsFile)));

            Console.WriteLine($"{GetTimeStamp()} Done.");

            DateTime now = DateTime.Now;
            long index = 0;
            var results = new List<Tuple<double/*predicted*/, double /*actual*/>>();
            //Parallel.ForEach(testRatings.GetUsers().TakeWhile(x => Interlocked.Read(ref index) < 5000), userId =>
            Parallel.ForEach(testRatings.GetUsers(), userId =>
            {
                if (Interlocked.Add(ref index, 1) % 100 == 0)
                    PrintProgress(index, now, testRatings.GetUsersCount());

                var ratings = testRatings.GetUserRatings(userId);
                foreach (var movieRating in ratings)
                {
                    double actual = movieRating.Value;
                    double predicted = trainingRatings.PredictMovieRating(userId, movieRating.Key, neighbors);
                    lock (results)
                        results.Add(new Tuple<double, double>(predicted, actual));
                }
            });

            PrintProgress(index, now, testRatings.GetUsersCount());

            double mae = CalculateMAE(results);
            double mape = CalculateMAPE(results);
            double rmsd = CalculateRMSD(results);
            Console.WriteLine();
            Console.WriteLine($"MAE={mae} from {index} users (Mean Absolute Error)");
            Console.WriteLine($"MAPE={mape} from {index} users (Mean Absolute Percentage Error)");
            Console.WriteLine($"RMSD={rmsd} from {index} users (Root-mean-square deviation)");
        }

        static void PrintProgress(long index, DateTime now, int totals)
        {
            Console.Write(
                $"\r[{(int)(DateTime.Now - now).TotalSeconds} seconds] Processed {index}/{totals} " +
                $"{100 * index / totals}%");
        }

        static double CalculateMAPE(List<Tuple<double, double>> results) => results.Sum(x => Math.Abs(x.Item2 - x.Item1) / x.Item2) / results.Count;

        static double CalculateRMSD(List<Tuple<double, double>> results) => Math.Sqrt(results.Sum(x => Math.Pow(x.Item1 - x.Item2, 2)) / results.Count);

        static double CalculateMAE(List<Tuple<double, double>> results) => results.Sum(x => Math.Abs(x.Item1 - x.Item2)) / results.Count;

        static string GetTimeStamp()
        {
            return $"[{DateTime.Now.ToString("HH:mm:ss.fff")}]";
        }

        static RatingData ParseRatingData(string[] lines)
        {
            var movieIds = new int[lines.Length];
            var userIds = new int[lines.Length];
            var ratings = new double[lines.Length];

            int index = 0;
            foreach (var line in lines.AsParallel())
            {
                string[] values = line.Split(new[] { "," }, StringSplitOptions.RemoveEmptyEntries);
                movieIds[index] = int.Parse(values[0]);
                userIds[index] = int.Parse(values[1]);
                ratings[index] = double.Parse(values[2]);
                index++;
            }

            return new RatingData(movieIds, userIds, ratings);
        }

        static MovieData ParseMoviesData(string[] lines)
        {
            var ids = new int[lines.Length];
            var years = new int[lines.Length];
            var titles = new string[lines.Length];

            int index = 0;
            foreach (var line in lines.AsParallel())
            {
                string[] values = line.Split(new[] { "," }, StringSplitOptions.RemoveEmptyEntries);

                ParseValue(values, ids, 0, index);
                ParseValue(values, years, 1, index);
                titles[index++] = string.Join(",", values, 2, values.Length - 2);
            }

            return new MovieData(ids, years, titles);
        }

        static void ParseValue(string[] values, int[] array, int position, int index)
        {
            int value;
            if (int.TryParse(values[position], out value))
                array[index] = value;
            else
                array[index] = GetNullValue();
        }

        static int GetNullValue()
        {
            //Console.WriteLine("NullValue");
            return 2000; // Hardcoded value
        }

        public class MovieData
        {
            readonly int[] ids;
            readonly int[] years;
            readonly string[] titles;

            public MovieData(int[] ids, int[] years, string[] titles)
            {
                this.ids = ids;
                this.years = years;
                this.titles = titles;
            }

            public Movie this[int index] => new Movie(index, this);

            public class Movie
            {
                readonly MovieData parent;
                readonly int index;

                public Movie(int index, MovieData parent)
                {
                    this.index = index;
                    this.parent = parent;
                }

                public int? Id => parent.ids[index];
                public int? Year => parent.years[index];
                public string Title => parent.titles[index];
            }
        }

        public class RatingData
        {
            readonly ConcurrentDictionary<int /*userId*/, double /*meanRating*/> userMeans;
            readonly ConcurrentDictionary<int /*userId*/, Dictionary<int, double>> userIndex;
            readonly Dictionary<int /*movieId*/, HashSet<int/*userId*/>> movieIndex;

            public RatingData(int[] movieIds, int[] userIds, double[] ratings)
            {
                userMeans = new ConcurrentDictionary<int, double>();
                userIndex = BuildUserIndex(movieIds, userIds, ratings);
                movieIndex = BuildMovieIndex(movieIds, userIds);
            }

            public double CalculateMeanVote(int userId)
            {
                double userMean;
                if (!userMeans.TryGetValue(userId, out userMean))
                {
                    Dictionary<int, double> ratings = GetUserRatings(userId);
                    userMean = ratings.Values.Sum() / ratings.Count;
                    userMeans.TryAdd(userId, userMean);
                }
                return userMean;
            }

            public double PredictMovieRating(int activeUserId, int movieId, int kNeighbors, double minimumDistance = 30)
            {
                var userRatings = GetUserRatings(activeUserId);
                double score;
                if (userRatings.TryGetValue(movieId, out score))
                    return score;

                double meanA = CalculateMeanVote(activeUserId);
                var neighbors = GetUserNeighbors(activeUserId, movieId, kNeighbors);

                double prediction = 0.0d;
                double absWeightSum = double.Epsilon; // Use this value to avoid a 0 denominator; it is insignificant
                foreach (var userId in neighbors)
                {
                    double weight = CalculateWeight(activeUserId, userId);
                    absWeightSum += Math.Abs(weight);
                    prediction += weight * (GetUserRatings(userId)[movieId] - CalculateMeanVote(userId));
                }

                prediction /= absWeightSum; // K normalization factor
                prediction += meanA;
                return NormalizedRange(prediction);
            }

            double NormalizedRange(double prediction)
            {
                return prediction < 1 ? 1 : (prediction > 5 ? 5 : prediction);
            }

            List<int> GetUserNeighbors(int activeUserId, int movieId, int k)
            {
                var nearestNeighbors = new List<int>();
                //foreach (var userId in movieIndex[movieId].OrderByDescending(uId => GetDistance(activeUserId, uId)))
                foreach (var userId in movieIndex[movieId])
                {
                    if (userId == activeUserId) continue;

                    //if (KNNs.Count < k && GetDistance(activeUserId, userId) >= distance)
                    if (nearestNeighbors.Count < k)
                        nearestNeighbors.Add(userId);
                }

                return nearestNeighbors;
            }

            // ReSharper disable once UnusedMember.Local
            double GetDistance(int fromUserId, int toUserId) => GetIntersection(fromUserId, toUserId).Count;

            public double CalculateWeight(int activeUserId, int toUserId)
            {
                double meanA = CalculateMeanVote(activeUserId);
                double meanI = CalculateMeanVote(toUserId);

                var commonRatings = GetIntersection(activeUserId, toUserId);

                double numerator = commonRatings.Sum(x => (x.Item2 - meanA) * (x.Item3 - meanI));
                double denominator = Math.Sqrt(commonRatings.Sum(x => Math.Pow(x.Item2 - meanA, 2) * Math.Pow(x.Item3 - meanI, 2)));

                return Math.Abs(denominator) < double.Epsilon ? 0 : numerator / denominator;
            }

            ConcurrentDictionary<int, Dictionary<int, double>> BuildUserIndex(int[] movieIds, int[] userIds, double[] ratings)
            {
                var localUserIndex = new ConcurrentDictionary<int, Dictionary<int, double>>();
                for (int index = 0; index < userIds.Length; index++)
                {
                    var userId = userIds[index];
                    Dictionary<int, double> movieRatings;
                    if (!localUserIndex.TryGetValue(userId, out movieRatings))
                    {
                        movieRatings = new Dictionary<int, double>();
                        localUserIndex.TryAdd(userId, movieRatings);
                    }

                    movieRatings.Add(movieIds[index], ratings[index]);
                }

                return localUserIndex;
            }
            Dictionary<int, HashSet<int>> BuildMovieIndex(int[] movieIds, int[] userIds)
            {
                var movieIndexLocal = new Dictionary<int /*movieId*/, HashSet<int> /*userIds*/>();
                for (int index = 0; index < movieIds.Length; index++)
                {
                    HashSet<int> users;
                    if (!movieIndexLocal.TryGetValue(movieIds[index], out users))
                    {
                        users = new HashSet<int>();
                        movieIndexLocal[movieIds[index]] = users;
                    }
                    users.Add(userIds[index]);
                }
                return movieIndexLocal;
            }

            public struct MovieRating
            {
                public int MovieId { get; }
                public double Rating { get; }

                public MovieRating(int movieId, double rating)
                {
                    MovieId = movieId;
                    Rating = rating;
                }
            }

            public IEnumerable<int /*userId*/> GetUsers() => userIndex.Select(kvp => kvp.Key);

            public int GetUsersCount() => userIndex.Count;

            public Dictionary<int/*movieId*/, double/*rating*/> GetUserRatings(int userId)
            {
                Dictionary<int, double> userRatings;
                userIndex.TryGetValue(userId, out userRatings);
                return userRatings;
            }

            public List<Tuple<int/*movieId*/, double/*lUser*/, double/*rUser*/>> GetIntersection(int lUser, int rUser)
            {
                var result = new List<Tuple<int /*movieId*/, double /*lUser*/, double /*rUser*/>>();

                var lUserRatings = GetUserRatings(lUser);
                var rUserRatings = GetUserRatings(rUser);
                Debug.Assert(lUserRatings != null);
                Debug.Assert(rUserRatings != null);

                foreach (var kvp in lUserRatings)
                {
                    double rUserRating;
                    if (rUserRatings.TryGetValue(kvp.Key, out rUserRating))
                        result.Add(new Tuple<int, double, double>(kvp.Key, kvp.Value, rUserRating));
                }

                return result;
            }
        }
    }
}
