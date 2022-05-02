/*
 * Anserini: A Lucene toolkit for replicable information retrieval research
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.anserini.ann;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import io.anserini.analysis.AnalyzerUtils;
import io.anserini.ann.fw.FakeWordsEncoderAnalyzer;
import io.anserini.ann.lexlsh.LexicalLshAnalyzer;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.queries.CommonTermsQuery;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopScoreDocCollector;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.OptionHandlerFilter;
import org.kohsuke.args4j.ParserProperties;

import static org.apache.lucene.search.BooleanClause.Occur.SHOULD;

public class ApproximateNearestNeighborMsMarcoEval {
  private static final String FW = "fw";
  private static final String LEXLSH = "lexlsh";

  public static final class Args {
    @Option(name = "-qv", metaVar = "[file]", required = true, usage = "query vectors model")
    public File queryVectors;

    @Option(name = "-path", metaVar = "[path]", required = true, usage = "index path")
    public Path path;

    @Option(name = "-qids", metaVar = "[file]", required = true, usage = "path to query ids file")
    public Path qids;

    @Option(name = "-encoding", metaVar = "[word]", required = true, usage = "encoding must be one of {fw, lexlsh}")
    public String encoding;

    @Option(name = "-depth", metaVar = "[int]", usage = "retrieval depth")
    public int depth = 1000;

    @Option(name = "-samples", metaVar = "[int]", usage = "no. of samples")
    public int samples = Integer.MAX_VALUE;

    @Option(name = "-lexlsh.n", metaVar = "[int]", usage = "n-grams")
    public int ngrams = 2;

    @Option(name = "-lexlsh.d", metaVar = "[int]", usage = "decimals")
    public int decimals = 1;

    @Option(name = "-lexlsh.hsize", metaVar = "[int]", usage = "hash set size")
    public int hashSetSize = 1;

    @Option(name = "-lexlsh.h", metaVar = "[int]", usage = "hash count")
    public int hashCount = 1;

    @Option(name = "-lexlsh.b", metaVar = "[int]", usage = "bucket count")
    public int bucketCount = 300;

    @Option(name = "-fw.q", metaVar = "[int]", usage = "quantization factor")
    public int q = 60;

    @Option(name = "-cutoff", metaVar = "[float]", usage = "tf cutoff factor")
    public float cutoff = 0.9999999f;

    @Option(name = "-msm", metaVar = "[float]", usage = "minimum should match")
    public float msm = 0;

    @Option(name = "-output", metaVar = "[file]", required = true, usage = "Output run file.")
    public String output = "out.txt";
  }

  public static void main(String[] args) throws Exception {
    ApproximateNearestNeighborMsMarcoEval.Args indexArgs = new ApproximateNearestNeighborMsMarcoEval.Args();
    CmdLineParser parser = new CmdLineParser(indexArgs, ParserProperties.defaults().withUsageWidth(90));

    try {
      parser.parseArgument(args);
    } catch (CmdLineException e) {
      System.err.println(e.getMessage());
      parser.printUsage(System.err);
      System.err.println("Example: " + ApproximateNearestNeighborMsMarcoEval.class.getSimpleName() +
          parser.printExample(OptionHandlerFilter.REQUIRED));
      return;
    }
    Analyzer vectorAnalyzer;
    if (indexArgs.encoding.equalsIgnoreCase(FW)) {
      vectorAnalyzer = new FakeWordsEncoderAnalyzer(indexArgs.q);
    } else if (indexArgs.encoding.equalsIgnoreCase(LEXLSH)) {
      vectorAnalyzer = new LexicalLshAnalyzer(indexArgs.decimals, indexArgs.ngrams, indexArgs.hashCount,
                                              indexArgs.bucketCount, indexArgs.hashSetSize);
    } else {
      parser.printUsage(System.err);
      System.err.println("Example: " + ApproximateNearestNeighborMsMarcoEval.class.getSimpleName() +
          parser.printExample(OptionHandlerFilter.REQUIRED));
      return;
    }

    System.out.println(String.format("Loading model %s", indexArgs.queryVectors));

    Map<String, List<float[]>> wordVectors = IndexVectors.readGloVe(indexArgs.queryVectors);

    Path indexDir = indexArgs.path;
    if (!Files.exists(indexDir)) {
      Files.createDirectories(indexDir);
    }

    System.out.println(String.format("Reading index at %s", indexArgs.path));

    Directory d = FSDirectory.open(indexDir);
    DirectoryReader reader = DirectoryReader.open(d);
    IndexSearcher searcher = new IndexSearcher(reader);
    if (indexArgs.encoding.equalsIgnoreCase(FW)) {
      searcher.setSimilarity(new ClassicSimilarity());
    }

    double time = 0d;

    PrintWriter out = new PrintWriter(Files.newBufferedWriter(Paths.get(indexArgs.output), StandardCharsets.UTF_8));
    int queryCount = 0;
    List<String> lines = FileUtils.readLines(indexArgs.qids.toFile(), "utf-8");
    for (String line : lines) {
      String[] split = line.trim().split("\t");
      String qid = split[0];
      String query = split[1];

      if (wordVectors.containsKey(qid)) {
        try {
          List<float[]> vectors = wordVectors.get(qid);
          for (float[] vector : vectors) {
            StringBuilder sb = new StringBuilder();
            for (double fv : vector) {
              if (sb.length() > 0) {
                sb.append(' ');
              }
              sb.append(fv);
            }
            String fvString = sb.toString();

            CommonTermsQuery simQuery = new CommonTermsQuery(SHOULD, SHOULD, indexArgs.cutoff);
            if (indexArgs.msm > 0) {
              simQuery.setLowFreqMinimumNumberShouldMatch(indexArgs.msm);
            }
            for (String token : AnalyzerUtils.analyze(vectorAnalyzer, fvString)) {
              simQuery.add(new Term(IndexVectors.FIELD_VECTOR, token));
            }
            Query fquery = simQuery;
//            BooleanQuery.Builder builder = new BooleanQuery.Builder();
//            for (String token : AnalyzerUtils.analyze(vectorAnalyzer, fvString)) {
//              builder.add(new BooleanClause(new TermQuery(new Term(IndexVectors.FIELD_VECTOR, token)), SHOULD));
//            }
//            builder.setMinimumNumberShouldMatch(1);
//            Query fquery = builder.build();

            long start = System.currentTimeMillis();
            TopScoreDocCollector results = TopScoreDocCollector.create(indexArgs.depth, Integer.MAX_VALUE);
            searcher.search(fquery, results);
            time += System.currentTimeMillis() - start;
            System.out.println("Query " + qid + " - " + time);

            int rank = 0;
            for (ScoreDoc sd : results.topDocs().scoreDocs) {
              Document document = reader.document(sd.doc);
              String id = document.get(IndexVectors.FIELD_ID);
              out.println(query + "\t" + id + "\t" + (rank + 1));
              rank++;
            }
            out.flush();
            queryCount++;
          }
        } catch (IOException e) {
          System.err.println("search for '" + query + "' failed " + e.getLocalizedMessage());
        }
      }
      if (queryCount >= indexArgs.samples) {
        break;
      }
    }
    out.close();
    time /= queryCount;

    System.out.println(String.format("avg query time: %s ms", time));

    reader.close();
    d.close();
  }

}
