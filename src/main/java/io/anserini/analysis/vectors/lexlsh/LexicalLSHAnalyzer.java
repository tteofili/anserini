package io.anserini.analysis.vectors.lexlsh;

import io.anserini.analysis.vectors.FeatureVectorsTokenizer;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.minhash.MinHashFilter;
import org.apache.lucene.analysis.miscellaneous.RemoveDuplicatesTokenFilter;
import org.apache.lucene.analysis.shingle.ShingleFilter;

/**
 * {@link Analyzer} that "lexically quantize" input vectors by tagging vector components by feature index, trimming them
 * up to a configurable number of decimal places, eventually aggregate them using n-grams and finally generate a text
 * fingerprint via LSH.
 */
public class LexicalLSHAnalyzer extends Analyzer {

  private static final int DEFAULT_SHINGLE_SIZE = 5;
  private static final int DEFAULT_DECIMALS = 1;

  private final int min;
  private final int max;
  private final int hashCount;
  private final int bucketCount;
  private final int hashSetSize;
  private final int decimals;

  private LexicalLSHAnalyzer(int min, int max, int hashCount, int bucketCount, int hashSetSize, int decimals) {
    super();
    this.min = min;
    this.max = max;
    this.hashCount = hashCount;
    this.bucketCount = bucketCount;
    this.hashSetSize = hashSetSize;
    this.decimals = decimals;
  }

  public LexicalLSHAnalyzer() {
    this(DEFAULT_SHINGLE_SIZE, DEFAULT_SHINGLE_SIZE, MinHashFilter.DEFAULT_HASH_COUNT, MinHashFilter.DEFAULT_BUCKET_COUNT,
        MinHashFilter.DEFAULT_HASH_SET_SIZE, DEFAULT_DECIMALS);
  }

  public LexicalLSHAnalyzer(int decimals, int ngrams, int hashCount, int bucketCount, int hashSetSize) {
    this(ngrams, ngrams, decimals, hashCount, bucketCount, hashSetSize);
  }

  @Override
  protected TokenStreamComponents createComponents(String fieldName) {
    Tokenizer source = new FeatureVectorsTokenizer();
    TokenFilter truncate = new LexicalLshTruncateTokenFilter(source, decimals);
    TokenFilter featurePos = new LexicalLshFeaturePositionTokenFilter(truncate);
    TokenStream filter;
    if (min > 1) {
      ShingleFilter shingleFilter = new ShingleFilter(featurePos, min, max);
      shingleFilter.setTokenSeparator(" ");
      shingleFilter.setOutputUnigrams(false);
      shingleFilter.setOutputUnigramsIfNoShingles(false);
      filter = new MinHashFilter(shingleFilter, hashCount, bucketCount, hashSetSize, bucketCount > 1);
    } else {
      filter = new MinHashFilter(featurePos, hashCount, bucketCount, hashSetSize, bucketCount > 1);
    }
    return new TokenStreamComponents(source, new RemoveDuplicatesTokenFilter(filter));
  }

}