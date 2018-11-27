package com.jstarcraft.module.text;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.hamcrest.CoreMatchers;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.jstarcraft.module.recommendation.configure.Configuration;
import com.jstarcraft.module.recommendation.evaluator.ranking.AUCEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.MAPEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.MRREvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.NDCGEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.PrecisionEvaluator;
import com.jstarcraft.module.recommendation.evaluator.ranking.RecallEvaluator;
import com.jstarcraft.module.recommendation.task.RankingTask;

public class TFIDFTestCase {

	@Test
	public void testRecommender() throws Exception {
		Configuration configuration = Configuration.valueOf("rec/content/tfidf-test.properties");
		RankingTask job = new RankingTask(configuration);
		Map<String, Double> measures = job.execute();
		Assert.assertThat(measures.get(MAPEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.04555074555074555D));
		Assert.assertThat(measures.get(AUCEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.6239576213260425D));
		Assert.assertThat(measures.get(NDCGEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.08569096830551297D));
		Assert.assertThat(measures.get(PrecisionEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.02272727272727273D));
		Assert.assertThat(measures.get(RecallEvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.2196969696969697D));
		Assert.assertThat(measures.get(MRREvaluator.class.getSimpleName()), CoreMatchers.equalTo(0.04663299663299663D));
	}

	private List<Collection<String>> documents;

	private Set<String> document1Terms = ImmutableSet.of("to", "be", "or", "not", "to", "be", "to be or", "be or not", "or not to", "not to be");

	private Set<String> document2Terms = ImmutableSet.of("or", "to", "jump", "or to jump");

	@Before
	public void setUp() {
		documents = Lists.newArrayList(NgramTfIdf.ngramDocumentTerms(Lists.newArrayList(1, 3), Arrays.asList("to be or not to be", "or to jump")));
	}

	@Test
	public void tf() {
		Map<String, Double> tf;
		tf = TfIdf.tf(documents.get(0));
		Assert.assertEquals(document1Terms, tf.keySet());
		Assert.assertEquals((Double) 2.0, tf.get("to"));
		Assert.assertEquals((Double) 2.0, tf.get("be"));
		Assert.assertEquals((Double) 1.0, tf.get("or"));
		Assert.assertEquals((Double) 1.0, tf.get("not"));

		tf = TfIdf.tf(documents.get(1));
		Assert.assertEquals(document2Terms, tf.keySet());
		Assert.assertEquals((Double) 1.0, tf.get("or"));
		Assert.assertEquals((Double) 1.0, tf.get("to"));
		Assert.assertEquals((Double) 1.0, tf.get("jump"));
	}

	@Test
	public void idf() {
		Iterable<Map<String, Double>> tfs = TfIdf.tfs(documents);
		Map<String, Double> idf = TfIdf.idfFromTfs(tfs);
		Set<String> allTerms = new HashSet<>();
		allTerms.addAll(document1Terms);
		allTerms.addAll(document2Terms);
		Assert.assertEquals(allTerms, idf.keySet());
		Assert.assertEquals((Double) 1.0, idf.get("to"));
		Assert.assertEquals((Double) (1 + Math.log(3.0 / 2.0)), idf.get("be"));
		Assert.assertEquals((Double) 1.0, idf.get("or"));
		Assert.assertEquals((Double) (1 + Math.log(3.0 / 2.0)), idf.get("not"));
		Assert.assertEquals((Double) (1 + Math.log(3.0 / 2.0)), idf.get("jump"));
	}

	@Test
	public void tfIdf() {
		List<Map<String, Double>> tfs = Lists.newArrayList(TfIdf.tfs(documents));
		Map<String, Double> idf = TfIdf.idfFromTfs(tfs);
		Map<String, Double> tfIdf;

		tfIdf = TfIdf.tfIdf(tfs.get(0), idf);
		Assert.assertEquals(Sets.newHashSet(document1Terms), tfIdf.keySet());
		Assert.assertEquals((Double) 2.0, tfIdf.get("to"));
		Assert.assertEquals((Double) (2.0 * (1 + Math.log(3.0 / 2.0))), tfIdf.get("be"));
		Assert.assertEquals((Double) 1.0, tfIdf.get("or"));
		Assert.assertEquals((Double) (1 + Math.log(3.0 / 2.0)), tfIdf.get("not"));

		tfIdf = TfIdf.tfIdf(tfs.get(1), idf);
		Assert.assertEquals(Sets.newHashSet(document2Terms), tfIdf.keySet());
		Assert.assertEquals((Double) 1.0, tfIdf.get("or"));
		Assert.assertEquals((Double) 1.0, tfIdf.get("to"));
		Assert.assertEquals((Double) (1 + Math.log(3.0 / 2.0)), tfIdf.get("jump"));
	}

	@Test
	public void ngramDocumentTerms() {
		Assert.assertEquals(2, documents.size());

		Set<String> terms;
		terms = Sets.newHashSet(documents.get(0));
		Assert.assertEquals(document1Terms, terms);
		terms = Sets.newHashSet(documents.get(1));
		Assert.assertEquals(document2Terms, terms);
	}

}
