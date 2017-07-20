package org.apache.lucene.search.similarities;

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.apache.lucene.index.FieldInvertState;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.SmallFloat;

/**
 * Expert: Default scoring implementation which {@link #encodeNormValue(float)
 * encodes} norm values as a single byte before being stored. At search time,
 * the norm byte value is read from the index
 * {@link org.apache.lucene.store.Directory directory} and
 * {@link #decodeNormValue(long) decoded} back to a float <i>norm</i> value.
 * This encoding/decoding, while reducing index size, comes with the price of
 * precision loss - it is not guaranteed that <i>decode(encode(x)) = x</i>. For
 * instance, <i>decode(encode(0.89)) = 0.75</i>.
 * <p>
 * Compression of norm values to a single byte saves memory at search time,
 * because once a field is referenced at search time, its norms - for all
 * documents - are maintained in memory.
 * <p>
 * The rationale supporting such lossy compression of norm values is that given
 * the difficulty (and inaccuracy) of users to express their true information
 * need by a query, only big differences matter. <br>
 * &nbsp;<br>
 * Last, note that search time is too late to modify this <i>norm</i> part of
 * scoring, e.g. by using a different {@link Similarity} for search.
 *
 * 参考<a>http://www.cnblogs.com/forfuture1978/archive/2010/02/08/1666137.html</a>
 * 
 * lucene的打分公式：
 *
 *  score(q,d)   =   (6)coord(q,d)  *  (3)queryNorm(q)  * ∑( (4)tf(t in d)  *  (5)idf(t)2  *  t.getBoost() *  (1)norm(t,d) )
 *  t in q
 *
 *  norm(t,d)   =   doc.getBoost()  *  (2)lengthNorm(field)  *  ∏f.getBoost()
 *  field f in d named as t
 *
 * lucene打分每一步对应的方法如下：
 *  (1) float computeNorm(String field, FieldInvertState state)
 *  (2) float lengthNorm(String fieldName, int numTokens)
 *  (3) float queryNorm(float sumOfSquaredWeights)
 *  (4) float tf(float freq)
 *  (5) float idf(int docFreq, int numDocs)
 *  (6) float coord(int overlap, int maxOverlap)
 *  (7) float scorePayload(int docId, String fieldName, int start, int end, byte [] payload, int offset, int length)
 *
 */
public class DefaultSimilarity extends TFIDFSimilarity {
  
  /** Cache of decoded bytes. */
  private static final float[] NORM_TABLE = new float[256];

  static {
    for (int i = 0; i < 256; i++) {
      NORM_TABLE[i] = SmallFloat.byte315ToFloat((byte)i);
    }
  }

  /** Sole constructor: parameter-free */
  public DefaultSimilarity() {}
  
  /** Implemented as <code>overlap / maxOverlap</code>. */
  @Override
  public float coord(int overlap, int maxOverlap) {
    return overlap / (float)maxOverlap;
  }

  /** Implemented as <code>1/sqrt(sumOfSquaredWeights)</code>.
   *
   * 这是按照向量空间模型，对query向量的归一化。此值并不影响排序，而仅仅使得不同的query之间的分数可以比较
   */
  @Override
  public float queryNorm(float sumOfSquaredWeights) {
    return (float)(1.0 / Math.sqrt(sumOfSquaredWeights));
  }
  
  /**
   * Encodes a normalization(标准) factor(因子) for storage in an index.
   * <p>
   * The encoding uses a three-bit mantissa, a five-bit exponent, and the
   * zero-exponent point at 15, thus representing values from around 7x10^9 to
   * 2x10^-9 with about one significant decimal digit of accuracy. Zero is also
   * represented. Negative numbers are rounded up to zero. Values too large to
   * represent are rounded down to the largest representable value. Positive
   * values too small to represent are rounded up to the smallest positive
   * representable value.
   * 
   * @see org.apache.lucene.document.Field#setBoost(float)
   * @see org.apache.lucene.util.SmallFloat
   */
  @Override
  public final long encodeNormValue(float f) {
    return SmallFloat.floatToByte315(f);
  }

  /**
   * Decodes the norm value, assuming(假设) it is a single byte.
   * 
   * @see #encodeNormValue(float)
   */
  @Override
  public final float decodeNormValue(long norm) {
    return NORM_TABLE[(int) (norm & 0xFF)];  // & 0xFF maps negative bytes to positive above 127
  }

  /** Implemented as
   *  <code>state.getBoost()*lengthNorm(numTerms)</code>, where
   *  <code>numTerms</code> is {@link FieldInvertState#getLength()} if {@link
   *  #setDiscountOverlaps} is false, else it's {@link
   *  FieldInvertState#getLength()} - {@link
   *  FieldInvertState#getNumOverlap()}.
   *
   *  主要计算文档长度的归一化，默认是 "1.0 / Math.sqrt(numTerms)", 因为默认的时候state.getBoost()=1.0;
   *
   *  为什么需要做归一化处理？因为有的文档较长，出现某个词的数目(即freq)较大，得分较高；而小文档的freq较小得分较小，但词出现的比例小文档比长文档更大，
   *  所以这样对小文档是不公平的，所以此处除以文档的长度，以便减少因文档长度造成的打分不公；至于为什么要开平方根 Math.sqrt(numTerms)？这是更深
   *  层次的考虑，此处不做深究
   *
   *  参考：<a>http://www.cnblogs.com/forfuture1978/archive/2010/02/08/1666137.html</a>
   *
   * state.getBoost()  是由创建索引时指定的field权重
   * numTerms   代表term对应field的term总数（例如：title:lucene and solr分词之后是lucene|and|solr，那么numTerms就是3）
   * Math.sqrt(numTerms) 对numTerms求平方根
   *
   *  @lucene.experimental */
  @Override
  public float lengthNorm(FieldInvertState state) {
    final int numTerms;
    if (discountOverlaps)
      numTerms = state.getLength() - state.getNumOverlap();
    else
      numTerms = state.getLength();
   return state.getBoost() * ((float) (1.0 / Math.sqrt(numTerms)));
  }

  /** Implemented as <code>sqrt(freq)</code>.
   * freq是指在一篇文档中包含的某个词的数目。tf是根据此数目给出的分数，默认为Math.sqrt(freq)。也即此项并不是随着包含的数目的增多而线性增加的
   */
  @Override
  public float tf(float freq) {
    return (float)Math.sqrt(freq);
  }
    
  /** Implemented as <code>1 / (distance + 1)</code>. */
  @Override
  public float sloppyFreq(int distance) {
    return 1.0f / (distance + 1);
  }
  
  /** The default implementation returns <code>1</code> */
  @Override
  public float scorePayload(int doc, int start, int end, BytesRef payload) {
    return 1;
  }

  /** Implemented as <code>log(numDocs/(docFreq+1)) + 1</code>.
   *
   *  Math.log(double a)  返回自然对数（以e为底）的一个double值
   *  根据包含某个词的文档数以及总文档数计算出的分数
   */
  @Override
  public float idf(long docFreq, long numDocs) {
    return (float)(Math.log(numDocs/(double)(docFreq+1)) + 1.0);
  }

  /** 
   * True if overlap tokens (tokens with a position of increment of zero) are
   * discounted from the document's length.
   */
  protected boolean discountOverlaps = true;

  /** Determines whether overlap tokens (Tokens with
   *  0 position increment) are ignored when computing
   *  norm.  By default this is true, meaning overlap
   *  tokens do not count when computing norms.
   *
   *  @lucene.experimental
   *
   *  @see #computeNorm
   */
  public void setDiscountOverlaps(boolean v) {
    discountOverlaps = v;
  }

  /**
   * Returns true if overlap tokens are discounted from the document's length. 
   * @see #setDiscountOverlaps 
   */
  public boolean getDiscountOverlaps() {
    return discountOverlaps;
  }

  @Override
  public String toString() {
    return "DefaultSimilarity";
  }
}
